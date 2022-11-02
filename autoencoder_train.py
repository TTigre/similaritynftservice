import json
from random import randint, random
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import utils
import autoencoder_models

# Generates the training set for the autoencoder using fake random NFTs
class CustomNFTGen(tf.keras.utils.Sequence):
    def __init__(self, batch_size=64, total_size=1000000):
        self.batch_size=batch_size
        self.total_size=total_size
        self.data=[]

    def __getitem__(self, index):
        if index<len(self.data):
            return self.data[index]
        fullarray=utils.NFTGenerate()
        for i in range(1,self.batch_size):
            fullarray=np.concatenate((fullarray,utils.NFTGenerate()),axis=0)
        self.data.append((fullarray,fullarray))
        return (fullarray,fullarray)

    def __len__(self):
        return self.total_size // self.batch_size

    def on_epoch_end(self):
        self.data.clear()    


encoder = autoencoder_models.encoder

autoencoder = autoencoder_models.autoencoder

# Generating the fake NFTs for training
traingen=CustomNFTGen(total_size=100000)

# Training the autoencoder
history = autoencoder.fit(traingen, batch_size=64, epochs=100)

# Saving the training results
encoder.save_weights("savedEncoder.json")