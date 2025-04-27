from medmnist import OCTMNIST
from sklearn.model_selection import train_test_split
import datasetClasse
from sklearn.model_selection import KFold
import numpy as np

#OrganSMNIST

ds_train=OCTMNIST(split='train', download=True, size=28)
ds_test=OCTMNIST(split='test', download=True, size=28)

workingDataset={}
testDataset={}
trainDataset={}
validationDataset={}

for name in range(4):
    workingDataset[name]=[]
    testDataset[name]=[]
    trainDataset[name]=[]
    validationDataset[name]=[]

for data, label in zip(ds_train.imgs, ds_train.labels):
    workingDataset[label[0]].append(data)
    
for data, label in zip(ds_test.imgs, ds_test.labels):
    testDataset[label[0]].append(data)
    
for name in range(4):
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    wds=np.array(workingDataset[name])
    for train_index, test_index in kf.split(wds):  
        trainDataset[name].append(wds[train_index])
        validationDataset[name].append(wds[test_index])

for name in range(4):
    newDataset=datasetClasse.datasetClasse(str(name), trainData=trainDataset[name], validationData=validationDataset[name], testData=testDataset[name])
    newDataset.saveDataset()

