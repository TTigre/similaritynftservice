# NFT Similarity Service
This repo implements some similarity algorithms for AlgoSeas NFTs

# The first idea

The first idea was to make an autoencoder that had close to 1 node as output of the encoder. This encoder would receive and output the NFTs of interests. If this approach worked it is supposed that 2 similar NFTs would have a similar encoder output.

Unfortunately the encoder had very poor results so the training dataset size was increased immensely with fake generated NFTs. There was an improvement in the results, but those improvements were not enough. It is a possibility more model training was needed but we practically achieved the limits of CPU training due to a lack of GPU.

# The follow up

The next idea was to use a model to predict NFT prices and according to this, search for more similarities in the set of similar prices. The encoder model was used to predict prices too because that way it is possible to do transfer learning. This model was more accurate, faster to train and the tests would prove its results were accurate enough for our purposes.

# The score function/heuristic

This alternative was conceived in parallel. It is a function that scores the similarity of 2 NFTs and admits weights for increasing or decreasing attribute importance. In the case of numeric values, the score decreases with the distance between the values. With categorical attributes if they have the same value the score increases by 1 or weight, if not it stays the same. We could say the score is the total categorical attributes with the same value plus the total numeric attributes minus the total numeric value distance (tCA+tNA-tNVd). It was sandboxed and it usually aligns with logical intuition.

# The final heuristics

In the end there were 2 heuristics:

## Heuristic 1: Encoder + Score

The encoder is used to predict the price of the NFT of interest and that way we select the NFTs with closest price and sort them according to the score function with the NFT of interest. It usually takes around 50 ms to execute in the included example, in a 8th gen mobile core i5 CPU. The execution time could improve with batching or with a GPU.

## Heuristic 2: Score

In this case the score function is calculated of the NFT of interest with all the possible similar NFTs and the list is sorted by the score. It is simple and efficient. It executed in approximately 10 ms with the same hardware than the previous heuristic. These results could be improved by code optimization.

## Beyond

Both heuristics weights can be tuned to sort the results, but the default weights would produce the most similar anyways between the firsts results. The first model could be better at discarding irrelevant elements and the span could be regulated to increase or decrease.

# Encoder structure and possible improvements

The encoder is a simple deep neural network of dense layers. Some hyperparameters could be tuned to improve the model results, also the train dataset size could be increased or some dropout or other kinds of layers could be added. This simplicity allows it to execute in a relatively small ammount of time at the CPU. I think it could be better too with a GPU, unfortunately the access to this hardware was not possible.

# File structure

## autoencoder_models.py

This file contains the autoencoder models. The encoder and decoder.

## autoencoder_train.py

If executed this file trains the autoencoder with the fake NFTs and saves the results in savedEncoder.json

## encoder_train.py

If executed this file loads the savedEncoder.json weights and trains the encoder in price prediction with a list of 475 price listings and saves the results in priceEncoder.json

## score_encoder_heuristic.py

This file contains a method that allows to run heuristic 1, at the end there is commented code to test the execution time, and an example

## score_heuristic.py

This file contains a method that allows to run heuristic 2, at the end there is commented code to test the execution time, and an example

## utils.py

Several important methods and utilities for the whole project. Here is saved the score function between others