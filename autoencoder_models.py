import json
from random import randint, random
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import utils

# This file contains the autoencoder used to pretrain the encoder part for later usage

# Encoder part of the autoencoder, later retrained to predict prices
input=keras.Input(shape=(173,))
encoderDense1 = layers.Dense(120, activation="relu")(input)
encoderDense2 = layers.Dense(70, activation="relu")(encoderDense1)
encoderDense3 = layers.Dense(30, activation="relu")(encoderDense2)
encoderDense4 = layers.Dense(1, activation="linear")(encoderDense3)

# Decoder part of the autoencoder
decoderDense1 = layers.Dense(30, activation="relu")(encoderDense4)
decoderDense2 = layers.Dense(70, activation="relu")(decoderDense1)
decoderDense3 = layers.Dense(120, activation="relu")(decoderDense2)
decoderDense4 = layers.Dense(120, activation="relu")(decoderDense3)
labelsoutput = layers.Dense(169, activation="sigmoid")(decoderDense4)
numericoutput = layers.Dense(4, activation="linear")(decoderDense4)
output = layers.concatenate([numericoutput,labelsoutput,])


encoder = keras.Model(inputs=input, outputs=encoderDense4, name="3levelEncoder")

autoencoder = keras.Model(inputs=input, outputs=output, name="3levelAutoencoder")

autoencoder.compile(
    loss=keras.losses.MeanSquaredError(),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["binary_accuracy","accuracy"],
)