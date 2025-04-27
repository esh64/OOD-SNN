from pytorch_ood.model import LeNet
from pytorch_ood.detector import MCD
from pytorch_ood.utils import OODMetrics
import Library
import numpy as np
import argparse
import torch
from torch.utils.data import TensorDataset, DataLoader

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--fold', type=int, default=0, metavar='N', help='input fold number (default: 0)')
parser.add_argument('--NoveltyDataset', type=str, default='', metavar='N', help='input OOD dataset MNIST, Blur, Noise, Rotate, Turn')

args = parser.parse_args()

if args.NoveltyDataset=='None':
    args.NoveltyDataset=''

detectorMethod='MCD'

kwargs = {'batch_size': 64}
kwargs.update({'num_workers': 1,
                'pin_memory': True,
                'shuffle': True},
                )

#loading dataset
testDataX=None
testDataY=[]

for name in range(3):
    classDataset=Library.importDataset(str(name))
    if type(testDataX)==type(None):
        testDataX=np.array(classDataset.testData)
        testDataY+=[1]*len(classDataset.testData)
    else:
        testDataX=np.concatenate((testDataX, np.array(classDataset.testData)), axis=0)
        testDataY+=[1]*len(classDataset.testData)

if args.NoveltyDataset=='':
    classDataset=Library.importDataset('Novelty')
    
elif args.NoveltyDataset!='':
    classDataset=Library.importDataset('Novelty_'+args.NoveltyDataset)
testDataX=np.concatenate((testDataX, np.array(classDataset.testData)), axis=0)
testDataY+=[-1]*len(classDataset.testData)

if testDataX.ndim == 3:
    testDataX = np.expand_dims(testDataX, axis=-1)
testDataX=np.moveaxis(testDataX, -1, 1)
testDataY=np.array(testDataY)
tensorTrainDataX = torch.from_numpy(testDataX).float() / 255.0
tensorTrainDataY = torch.from_numpy(testDataY).long()
dataset=TensorDataset(tensorTrainDataX.float(), tensorTrainDataY.float())
test_loader = torch.utils.data.DataLoader(dataset, **kwargs)

# Create Neural Network
model = LeNet(pretrained="fmnist_lenet_"+str(args.fold)+".pth").eval().cuda()

# Create detector
detector = MCD(model)

# Evaluate
metrics = OODMetrics()

for x, y in test_loader:
    metrics.update(detector(x.cuda()), y)

metricsValues=metrics.compute()

import csv
from os.path import exists

data = [args.fold, metricsValues['AUROC'], metricsValues['AUPR-IN'],metricsValues['FPR95TPR']]
if exists(detectorMethod+args.NoveltyDataset+'.csv'):
    with open(detectorMethod+args.NoveltyDataset+'.csv', 'a', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(data)
else:
    header = ['fold', 'AUCROC', 'AUPRC','FPR95TPR']
    with open(detectorMethod+args.NoveltyDataset+'.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow(data)

