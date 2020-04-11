# -*- coding: utf-8 -*-



from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)

import jax
import flax
from flax import jax_utils
from flax import nn
from flax import optim
from flax.training import common_utils

import numpy as np
import jax.numpy as jnp
import tensorflow as tf
import tensorflow_datasets as tfds
import collections
import functools

""" Data input pipeline written in tensorflow """

def vocab(path='tiny-shakespeare.txt'):
    file = open(path, 'r')
    data = file.read()
    freq = collections.Counter(data).most_common()
    vocab = dict()
    reverse_vocab = dict()
    count = 0
    for i in freq:
        vocab[count] = i[0]
        reverse_vocab[i[0]] = count
        count = count + 1
    return vocab, reverse_vocab


def get_text_dataset(text, reverse_vocab, mode, sequence_length=50, batch_size=32):
    reverse_list = list()
    for i in text:
        reverse_list.append(reverse_vocab[i])
    ds_seq = tf.data.Dataset.from_tensor_slices(tf.one_hot(reverse_list, depth=len(reverse_vocab)))
    ds_seq = ds_seq.batch(sequence_length, drop_remainder=True)
    ds = ds_seq.map(lambda x: chunk(x))
    if mode is tf.estimator.ModeKeys.TRAIN:
        ds.shuffle(len(text))
    ds = ds.batch(batch_size, drop_remainder=True)
    return ds

def chunk(x):
    return x[:-1], x[1:]    
        
def test_ds(vocab_size):
    start_id = np.random.randint(low=0, high=vocab_size, size=(1,1))
    test_ds = tf.data.Dataset.from_tensor_slices(tensors=tf.convert_to_tensor(start_id, dtype=tf.int64))
    test_ds = test_ds.map(map_func=lambda x:tf.one_hot(x, depth=vocab_size))
    return test_ds

f = open('tiny-shakespeare.txt', 'r')
text = f.read()
f.close()

vocab, reverse_vocab = vocab()
params = {'batch_size': 32, 'seq_length' : 50, 'learning_rate' : 0.002, 'epochs_per_decay': 5, 'learning_rate_decay' : 0.97, 'vocab_length' : len(vocab)}
params['step_decay'] = params['epochs_per_decay'] * int( int( len(text) / params['seq_length']) / params['batch_size'])
ds = get_text_dataset(text=text, reverse_vocab=reverse_vocab, mode=tf.estimator.ModeKeys.TRAIN)

""" The Flax RNN impementation """

class RNN(flax.nn.Module):
    """LSTM"""
    def apply(self, carry, inputs):
        carry1, outputs = jax_utils.scan_in_dim(
            nn.LSTMCell.partial(name='lstm1'), carry[0], inputs, axis=1)
        carry2, outputs = jax_utils.scan_in_dim(
            nn.LSTMCell.partial(name='lstm2'), carry[1], outputs, axis=1)
        carry3, outputs = jax_utils.scan_in_dim(
            nn.LSTMCell.partial(name='lstm3'), carry[2], outputs, axis=1)
        x = nn.Dense(outputs, features=params['vocab_length'], name='dense')
        return [carry1, carry2, carry3], x

class charRNN(flax.nn.Module):
    """Char Generator"""
    def apply(self, inputs, carry_pred=None, train=True):
        batch_size = params['batch_size']
        vocab_size = params['vocab_length']
        hidden_size = 512
        if train:
            carry1 = nn.LSTMCell.initialize_carry(jax.random.PRNGKey(0), (batch_size,),hidden_size)
            carry2 = nn.LSTMCell.initialize_carry(jax.random.PRNGKey(0), (batch_size,),hidden_size)
            carry3 = nn.LSTMCell.initialize_carry(jax.random.PRNGKey(0), (batch_size,),hidden_size)
            carry = [carry1, carry2, carry3]
            _, x = RNN(carry, inputs)
            return x
        else:
            carry, x = RNN(carry_pred, inputs)
            return carry, x
        


@jax.vmap
def cross_entropy_loss(logits, labels):
      """Returns cross-entropy loss."""
      return -jnp.mean(jnp.sum(nn.log_softmax(logits) * labels))

@jax.vmap
def acc(logits, labels):
      """Returns accuracy."""
      return jnp.argmax(logits, -1) == jnp.argmax(labels, -1)

def compute_metrics(logits, labels):
    """Computes metrics and returns them."""
    loss = jnp.mean(cross_entropy_loss(logits, labels)) / params['batch_size']
    
    accuracy = jnp.mean( acc(logits, labels)
        )
    metrics = {
        'loss': loss,
        'accuracy': accuracy,
    }
    return metrics


@jax.jit
def train_step(optimizer, batch):
    """Train one step."""
    
    def loss_fn(model):
        """Compute cross-entropy loss and predict logits of the current batch"""

        logits = model(batch[0])        
        loss = jnp.mean(cross_entropy_loss(logits, batch[1])) / params['batch_size']
        return loss, logits

    def exponential_decay(steps):
        """Decrease the learning rate every 5 epochs"""
        x_decay = (steps / params['step_decay']).astype('int32')
        ret = params['learning_rate']* jax.lax.pow((params['learning_rate_decay']), x_decay.astype('float32'))
        return jnp.asarray(ret, dtype=jnp.float32)

    current_step = optimizer.state.step
    new_lr = exponential_decay(current_step)
    # calculate and apply the gradient 
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, logits), grad = grad_fn(optimizer.target)
    new_optimizer = optimizer.apply_gradient(grad, learning_rate=new_lr)

    metrics = compute_metrics(logits, batch[1])
    metrics['learning_rate'] = new_lr
    return new_optimizer, metrics

@jax.jit
def sample(inputs, optimizer):
    next_inputs = inputs
    output = []
    batch_size = 1 
    carry1 = nn.LSTMCell.initialize_carry(jax.random.PRNGKey(0), (batch_size,),512)
    carry2 = nn.LSTMCell.initialize_carry(jax.random.PRNGKey(0), (batch_size,),512)
    carry3 = nn.LSTMCell.initialize_carry(jax.random.PRNGKey(0), (batch_size,),512)
    carry = [carry1, carry2, carry3]

    def inference(model, carry):
        carry, rnn_output = model(inputs=next_inputs, train=False, carry_pred=carry)
        return carry, rnn_output
  
    for i in range(200):
        carry, rnn_output = inference(optimizer.target, carry)
        output.append(jnp.argmax(rnn_output, axis=-1))
        # Select the argmax as the next input.
        next_inputs = jnp.expand_dims(common_utils.onehot(jnp.argmax(rnn_output), params['vocab_length']), axis=0)
    return output      


def create_model(rng):
    """Creates a model."""
    vocab_size = params['vocab_length']
    _, initial_params = charRNN.init_by_shape(
        rng, [((1, params['seq_length'], vocab_size), jnp.float32)])
    model = nn.Model(charRNN, initial_params)
    return model

def create_optimizer(model, learning_rate):
    """Creates an Adam optimizer for model."""
    optimizer_def = optim.Adam(learning_rate=learning_rate, weight_decay=1e-1)
    optimizer = optimizer_def.create(model)
    return optimizer

def train_model():
    """Train and inference """
    rng = jax.random.PRNGKey(0)
    model = create_model(rng)
    optimizer = create_optimizer(model, params['learning_rate'])

    del model
    for epoch in range(100):

        for text in tfds.as_numpy(ds):
            optimizer, metrics = train_step(optimizer, text)

        print('epoch: %d, loss: %.4f, accuracy: %.2f, LR: %.8f' % (epoch+1,metrics['loss'], metrics['accuracy'] * 100, metrics['learning_rate']))
        test = test_ds(params['vocab_length'])
        sampled_text = ""

        if ((epoch+1)%10 == 0):
            for i in test:
                sampled_text += vocab[int(jnp.argmax(i.numpy(),-1))]
                start = np.expand_dims(i, axis=0)
                text = sample(start, optimizer)

            for i in text:
                sampled_text += vocab[int(i)]
            print(sampled_text)

train_model()

