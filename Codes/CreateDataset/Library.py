import os
import scipy
import pickle
import datasetClasse


#creates datasets
def createValidationDict():
    dataset={}
    for name in range(10):
        fulldataset=importDataset(str(name))
        dataset[str(name)]=fulldataset.validationData
    return dataset

def createTestDict():
    dataset={}
    for name in range(10):
        fulldataset=importDataset(str(name))
        dataset[str(name)]=fulldataset.testData
    return dataset

#dataset
def importDataset(filename):
    local=os.getcwd()
    pos=local.find('OOD')
    if pos==-1:
        return None
    local=local[:pos+len('OOD')]+'/Dataset/'
    return pickle.load(open(local+str(filename)+'.dataset', "rb"))


