from cgitb import reset
import json
from random import randint, random
import pandas as pd
import numpy as np

# File with multiple utilities used along the project

f=open('savedattributes.json')

# Loading some sample data from a JSON file to know more about its structure
data = json.load(f)
dic={}
attributes={}

# The columns that have numeric value
numericValue=['combat','constitution','luck','plunder']

names=[]

# Categorical attributes and possible values
namesdic:dic={}
for k in data.keys():
    for k2 in data[k].keys():
        if k2 not in names and k2 not in numericValue:
            names.append(k2)
            dic[k2]=[]
            namesdic[k2]=['None',] # Adding the empty attribute option

for k in data.keys():
    for itemName in dic.keys():
        if itemName in data[k].keys():
            if itemName in numericValue:
                data[k][itemName]/=100.0  # Setting numeric values to a value in the range [0,100]
            elif data[k][itemName] not in namesdic[itemName]:
                namesdic[itemName].append(data[k][itemName]) # Adding pending attribute options

# Method that generates a numpy array for machine learning given the attribute dictionary of an NFT
def NFTNormalization(AttributeDictionary):
    resultList=[]
    for v in numericValue:
        resultList.append(AttributeDictionary[v]/100.0)
    for k in namesdic.keys():
        index=0
        if k in AttributeDictionary.keys():
            searchable=AttributeDictionary[k]
            if searchable==False:
                searchable='None'
            index=namesdic[k].index(searchable)
        for i in range(len(namesdic[k])):
            if i==index:
                resultList.append(1.0)
            else:
                resultList.append(0.0)
    return np.array(resultList).reshape((1,173))

# A similarity score beetwen 2 NFTs. The weights are dictionaries attributeName->weight. A proposed score function
def NFTSimilarityScore(NFT1,NFT2,numericweights=None,categoricalweights=None):
    score=0.0

    # Calculating numeric elements 1 - distance.
    for i in range(len(numericValue)):
        acum=(1.0-(abs(float(NFT1[numericValue[i]])-float(NFT2[numericValue[i]])))) # If they are equal the score is 1, else the score is proportional to the inverse of the distance
        if numericweights!=None:
            acum=acum*numericweights[numericValue[i]]
        score+=acum
    
    # Calculating categorical distance. If the attribute value is the same in both score+=1.
    for k in namesdic.keys():
        acum=0
        if k not in NFT1.keys() and k not in NFT2.keys():
            acum=1
        if k in NFT1.keys() and k in NFT2.keys():
            if NFT1[k]==NFT2[k]:
                acum=1
        if categoricalweights!=None:
            acum=acum*categoricalweights[k]
        score+=acum
    
    return score
    
# Generates a numpy array that represents a fake random NFT
def NFTGenerate():
    resultList=[]
    for v in numericValue:
        resultList.append(randint(0,100)/100.0)
    for k in namesdic.keys():
        index=randint(0,len(namesdic[k])-1)
        for i in range(len(namesdic[k])):
            if i==index:
                resultList.append(1.0)
            else:
                resultList.append(0.0)
    return np.array(resultList).reshape((1,173))

def GenerateAttributeDictionaryFromListing(ListingItem):
    result = ListingItem["assetInformation"]["nProps"]["properties"]
    result["price"] = ListingItem["assetInformation"]["listing"]["price"]
    result["id"]=ListingItem["assetInformation"]["SK"]
    return result

def ListingsParse():
    f=open('listings.json')

    listingData = json.load(f)

    f.close()

    listingsMaxPrice=0

    listingsList=[]
    for c in listingData:
        value=GenerateAttributeDictionaryFromListing(c)
        if value["price"]>listingsMaxPrice:
            listingsMaxPrice=value["price"]
        listingsList.append(value)

    f2=open('maxPrice.txt',"w")
    f2.writelines([listingsMaxPrice.__str__()])
    f2.close()
    
    return (listingsList,listingsMaxPrice)
