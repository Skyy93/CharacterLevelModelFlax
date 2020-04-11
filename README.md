# CharacterLevelModelFlax
Implementation of the Character Level Language Model in Flax


You have to install the following pip packages:


```
pip install -q --upgrade https://storage.googleapis.com/jax-releases/`nvcc -V | sed -En "s/.* release ([0-9]*)\.([0-9]*),.*/cuda\1\2/p"`/jaxlib-0.1.42-`python3 -V | sed -En "s/Python ([0-9]*)\.([0-9]*).*/cp\1\2/p"`-none-linux_x86_64.whl jax
  
pip install -q git+https://github.com/google/flax.git@master

pip install tensorflow

pip install tensorflow_datasets

```
