"""
reated on Tue Apr 14 12:49:09 2020

@author: benny
"""
import sys
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation, Flatten, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D, Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import random
#from tqdm import tqdm_notebook
import os
from keras.models import load_model


save_path = sys.argv[2]
if not os.path.isdir(sys.argv[2]):
    os.mkdir(sys.argv[2])

model_file=sys.argv[1]
g=load_model(model_file)
z_dim=100    
def plot_generated(n_ex=10, dim=(1, 10), figsize=(12, 2)):
    noise = np.random.normal(0, 1, size=(n_ex, z_dim))
    generated_images = g.predict(noise)
    generated_images = generated_images.reshape(generated_images.shape[0], 28, 28)
    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i, :, :], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()

    plt.savefig(f'{save_path}/gan-images.png')
    
    plt.show()
    
plot_generated()

