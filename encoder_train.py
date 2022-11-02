import json
from random import randint, random
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import autoencoder_models
import utils


# In this file the pretrained encoder is loaded and later trained using sales data to predict the price of NFTs

# Loading the pretrained encoder
encoder=autoencoder_models.encoder
encoder.load_weights("savedEncoder.json")

listings,maxprice=utils.ListingsParse()

# Preparing the train data
fullarray = utils.NFTNormalization(listings[0])
prices = np.array([listings[0]["price"]*1.0/maxprice,])

for i in range(1,len(listings)):
    fullarray=np.concatenate((fullarray,utils.NFTNormalization(listings[i])),axis=0)
    prices=np.concatenate((prices,np.array([listings[i]["price"]*1.0/maxprice,])),axis=0)
    
encoder.compile(
    loss=keras.losses.MeanSquaredError(),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["mean_squared_error"],
)
encoder.summary()

# Training the model with the sales data
history=encoder.fit(fullarray, prices, batch_size=64, epochs=100, validation_split=0.2)

# Saving the trained model
encoder.save_weights("priceEncoder.json")