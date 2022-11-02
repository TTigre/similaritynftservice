import json
from random import randint, random
from tabnanny import verbose
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import autoencoder_models
import utils
import score_heuristic
import timeit

# Reading the max price of the encoder
f=open("maxPrice.txt")
maxprice=int(f.readline())
f.close()

# Loading the trained encoder
encoder=autoencoder_models.encoder
encoder.load_weights("priceEncoder.json").expect_partial()

def PriceValue(NFT):
    return NFT["price"]

# Preparing the NFT list for binary search
NFTsList,_=utils.ListingsParse()
NFTsList.sort(key=PriceValue)

def binary_search(arr, low, high, x):
 
    # Check base case
    if high >= low:
 
        mid = (high + low) // 2
 
        # If element is present at the middle itself
        if arr[mid]["price"] == x:
            return mid
 
        # If element is smaller than mid, then it can only
        # be present in left subarray
        elif arr[mid]["price"] > x:
            return binary_search(arr, low, mid - 1, x)
 
        # Else the element can only be present in right subarray
        else:
            return binary_search(arr, mid + 1, high, x)
 
    else:
        # Element is not present in the array
        return high

# Predicting the NFT price with the encoder
def PredictNFTPrice(NFT):
    result=encoder.predict(utils.NFTNormalization(NFT),verbose=0)
    result=int(result*maxprice)
    return result
 
 #Using the price prediction to restrict the NFT List size, and then sorting by score heuristic
def GetClosePredictedPriceNFTs(NFT,NFTSortedList,span=50,numericweights=None,categoricalweights=None):
    price=PredictNFTPrice(NFT)
    elementIndex=binary_search(NFTSortedList,0,len(NFTSortedList)-1,price)
    lowerIndex=max(elementIndex-span,0)
    higherIndex=min(elementIndex+span,len(NFTSortedList))

    listaRetorno=[]
    for i in range(lowerIndex,higherIndex):
        listaRetorno.append(NFTSortedList[i])
    
    return score_heuristic.GetSimilarNFTsAndScore(NFT,listaRetorno,numericweights=numericweights,categoricalweights=categoricalweights)

numWeights={'combat':10.0,'constitution':10.0,'luck':1.0,'plunder':1.0}

# test_result=GetClosePredictedPriceNFTs(NFTsList[10],NFTsList,numericweights=numWeights)
# print(test_result)
    
# print(timeit.timeit(lambda: GetClosePredictedPriceNFTs(NFTsList[10],NFTsList,numericweights=numWeights),number=1000))