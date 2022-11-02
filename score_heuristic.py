from audioop import reverse
import utils
import timeit

# Score heuristic using the score function from utils. Sorts NFTs by similarity score.
def GetSimilarNFTsAndScore(NFT1,NFTList,numericweights=None,categoricalweights=None):
    def SortingFunction(NFT):
        return utils.NFTSimilarityScore(NFT1,NFT,numericweights, categoricalweights)
    
    workingList=NFTList.copy()
    workingList.sort(key=SortingFunction, reverse=True)

    return workingList

numWeights={'combat':10.0,'constitution':10.0,'luck':1.0,'plunder':1.0}

# listings,maxprice=utils.ListingsParse()
# scoreSorted=GetSimilarNFTsAndScore(listings[10],listings,numericweights=numWeights)
# print(scoreSorted)

# print(timeit.timeit(lambda: GetSimilarNFTsAndScore(listings[10],listings),number=1000))