from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import Library
import numpy as np
import argparse
from tqdm.auto import tqdm
import csv
from os.path import exists
import pickle

from sklearn.neighbors import NearestNeighbors
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from pyod.models.gmm import GMM
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder

class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(
        self, best_valid_loss=float('inf')
    ):
        self.best_valid_loss = best_valid_loss
        
    def __call__(
        self, current_valid_loss, 
        epoch, model, optimizer, criterion, name, fold
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, 'fmnist_lenet_'+str(name)+'_'+str(fold)+'.pth')

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        return x
    
def train(args, model, device, train_loader, optimizer, epoch, criterion):
    model.train()
    counter=0
    test_loss = 0
    for batch_idx, (anchor, positive, negative) in enumerate(train_loader):
        counter+=1
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
        optimizer.zero_grad()
        outputAnchor = model(anchor)
        outputPositive = model(positive)
        outputNegative = model(negative)
        loss = criterion(outputAnchor, outputPositive, outputNegative)
        loss.backward()
        test_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.36f}'.format(
                epoch, batch_idx * len(anchor), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break
            
    epoch_loss = test_loss / counter
    
    print('\nTest set: Average loss: {:.36f}\n'.format(epoch_loss))
    
    return epoch_loss

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    counter = 0
    with torch.no_grad():
        for anchor, positive, negative in test_loader:
            counter += 1
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            outputAnchor = model(anchor)
            outputPositive = model(positive)
            outputNegative = model(negative)
            test_loss += criterion(outputAnchor, outputPositive, outputNegative).item()  # sum up batch loss
            

    epoch_loss = test_loss / counter
    
    print('\nTest set: Average loss: {:.36f}\n'.format(epoch_loss))
    
    return epoch_loss

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Tensorflow FMNIST NoveltyDetection')
    parser.add_argument('--detectorClass', type=int, default=0, metavar='N',
                        help='input detector class class (default: 1)')
    parser.add_argument('--fold', type=int, default=0, metavar='N',
                        help='input fold number (default: 0)')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=125, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    #torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        kwargs.update({'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True},
                     )
        
    kwargs2 = {'batch_size': args.batch_size}
    if use_cuda:
        kwargs2.update({'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': False},
                     )

    classDataset={}
    testDataXReal=None
    testDataYReal=[]
    
    for name in range(3):
        classDataset[name]=Library.importDataset(name)
        if type(testDataXReal)==type(None):
            testDataXReal=np.array(classDataset[name].testData)
        else:
            testDataXReal=np.concatenate((testDataXReal, np.array(classDataset[name].testData)), axis=0)
        testDataYReal+=[1]*len(classDataset[name].testData)
        
    neigh_out_classe={}
    for name in range(3):
        if name != args.detectorClass:
            neigh_out_classe[name]=NearestNeighbors(n_neighbors=1, n_jobs=-1)
            newTrainData=np.array(classDataset[name].trainData[args.fold])
            newTrainData=newTrainData.reshape(-1, 28*28*1)
            neigh_out_classe[name].fit(newTrainData)
    
    #classe
    neigh_in_classe=NearestNeighbors(n_neighbors=2, n_jobs=-1)
    newTrainData=np.array(classDataset[args.detectorClass].trainData[args.fold])
    newTrainData=newTrainData.reshape(-1, 28*28*1)
    neigh_in_classe.fit(newTrainData)

    #montando o conjunto de treinamento
    train_dataset_anchor=[]
    train_dataset_positive=[]
    train_dataset_negative=[]
    for name in range(3):
        if name!=args.detectorClass:
            for positive_index, data, negative_index in zip(neigh_in_classe.kneighbors(newTrainData)[1], classDataset[args.detectorClass].trainData[args.fold], neigh_out_classe[name].kneighbors(newTrainData)[1]):
                train_dataset_anchor.append(data)
                train_dataset_positive.append(classDataset[args.detectorClass].trainData[args.fold][positive_index[-1]])
                train_dataset_negative.append(classDataset[name].trainData[args.fold][negative_index[0]])
    
    trainData_anchor=np.array(train_dataset_anchor)
    trainData_positive=np.array(train_dataset_positive)
    trainData_negative=np.array(train_dataset_negative)
    if trainData_anchor.ndim == 3:
        trainData_anchor = np.expand_dims(trainData_anchor, axis=-1)
        trainData_positive = np.expand_dims(trainData_positive, axis=-1)
        trainData_negative = np.expand_dims(trainData_negative, axis=-1)
    trainData_anchor=np.moveaxis(trainData_anchor, -1, 1)
    trainData_positive=np.moveaxis(trainData_positive, -1, 1)
    trainData_negative=np.moveaxis(trainData_negative, -1, 1)
    trainData_anchor = torch.from_numpy(trainData_anchor).float() / 255.0
    trainData_positive = torch.from_numpy(trainData_positive).float() / 255.0
    trainData_negative = torch.from_numpy(trainData_negative).float() / 255.0
    dataset1=TensorDataset(trainData_anchor, trainData_positive, trainData_negative)
    train_loader = torch.utils.data.DataLoader(dataset1,**kwargs)
    
    dataset1=TensorDataset(trainData_anchor, trainData_positive, trainData_negative)
    train_loader = torch.utils.data.DataLoader(dataset1,**kwargs)
    
    if testDataXReal.ndim == 3:
        testDataXReal = np.expand_dims(testDataXReal, axis=-1)
    testDataXReal=np.moveaxis(testDataXReal, -1, 1)
    testDataXReal = torch.from_numpy(testDataXReal).float() / 255.0
    testDataYReal = np.array(testDataYReal)
    testDataYReal = torch.from_numpy(testDataYReal).long()
    dataset=TensorDataset(testDataXReal, testDataYReal)
    test_loaderIND_real = torch.utils.data.DataLoader(dataset, **kwargs2)
    
    print("Datasets prontos")
    save_best_model = SaveBestModel()
    
    model = LeNet().to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    for epoch in range(1, args.epochs + 1):
        print("Epoch:" +str(epoch))
        epoch_loss=train(args, model, device, train_loader, optimizer, epoch, criterion)
        #epoch_loss=test(model, device, train_loader, criterion)
        scheduler.step(epoch_loss)
        save_best_model(epoch_loss, epoch, model, optimizer, criterion, args.detectorClass, args.fold)
        print('-'*10)
    
    #save_model(args.epochs, model, optimizer, criterion, args.detectorClass)
    print('TRAINING COMPLETE')
    
    best_model_cp = torch.load('fmnist_lenet_'+str(args.detectorClass)+'_'+str(args.fold)+'.pth', weights_only=False)
    model.load_state_dict(best_model_cp['model_state_dict'])
    
    print('TESTING')
    
    test_loaderOOD={}
    outputsOOD={}
    for novelty in ['']:
        if novelty=='':
            classDataset=Library.importDataset('Novelty')
        else:
            classDataset=Library.importDataset('Novelty_'+novelty)



        testDataOODX=np.array(classDataset.testData)
        testDataOODY=np.array([0]*len(classDataset.testData))
        if testDataOODX.ndim == 3:
            testDataOODX = np.expand_dims(testDataOODX, axis=-1)
        testDataOODX=np.moveaxis(testDataOODX, -1, 1)
        testDataOODX = torch.from_numpy(testDataOODX).float() / 255.0
        testDataOODY = np.array(testDataOODY)
        testDataOODY = torch.from_numpy(testDataOODY).long()
        dataset=TensorDataset(testDataOODX, testDataOODY)
        test_loaderOOD[novelty] = torch.utils.data.DataLoader(dataset, **kwargs2)
        outputsOOD[novelty]=None       
    
    detectorClass = Library.importDataset(str(args.detectorClass))

    # Treinamento
    trainDataX = detectorClass.trainData[args.fold]
    if trainDataX.ndim == 3:
        trainDataX = np.expand_dims(trainDataX, axis=-1)
    trainDataX = np.moveaxis(trainDataX, -1, 1)
    trainDataY = np.array([1] * len(trainDataX))

    # Validação
    validationDataX = detectorClass.validationData[args.fold]
    if validationDataX.ndim == 3:
        validationDataX = np.expand_dims(validationDataX, axis=-1)

    validationDataX = np.moveaxis(validationDataX, -1, 1)
    validationDataY = np.array([1] * len(validationDataX))

    # Teste IND
    testDataX = np.array(detectorClass.testData)
    if testDataX.ndim == 3:
        testDataX = np.expand_dims(testDataX, axis=-1)
    testDataX = np.moveaxis(testDataX, -1, 1)
    testDataY = np.array([1] * len(testDataX))

    # Conversão para tensores normalizados
    tensorTrainDataX = torch.from_numpy(trainDataX).float() / 255.0
    tensorTrainDataY = torch.from_numpy(trainDataY).long()

    tensorValidationDataX = torch.from_numpy(validationDataX).float() / 255.0
    tensorValidationDataY = torch.from_numpy(validationDataY).long()

    tensorTestDataINDX = torch.from_numpy(testDataX).float() / 255.0
    tensorTestDataINDY = torch.from_numpy(testDataY).long()

    # Criação dos datasets e dataloaders
    dataset1 = TensorDataset(tensorTrainDataX, tensorTrainDataY)
    dataset2 = TensorDataset(tensorValidationDataX, tensorValidationDataY)
    dataset3 = TensorDataset(tensorTestDataINDX, tensorTestDataINDY)

    train_loader = torch.utils.data.DataLoader(dataset1, **kwargs)
    valid_loader = torch.utils.data.DataLoader(dataset2, **kwargs)
    test_loaderIND = torch.utils.data.DataLoader(dataset3, **kwargs)
    
    model.eval()
    outputsINDTrain=None
    for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
        image, labels = data
        image = image.to(device)
        if type(outputsINDTrain)==type(None):
            outputsINDTrain=model(image).cpu().detach().numpy()
        else:
            outputsINDTrain=np.concatenate((outputsINDTrain, model(image).cpu().detach().numpy()), axis=0)
    
    outputsINDvalidation=None
    for i, data in tqdm(enumerate(valid_loader), total=len(valid_loader)):
        image, labels = data
        image = image.to(device)
        if type(outputsINDvalidation)==type(None):
            outputsINDvalidation=model(image).cpu().detach().numpy()
        else:
            outputsINDvalidation=np.concatenate((outputsINDvalidation, model(image).cpu().detach().numpy()), axis=0)
        
    outputsINDTest=None
    for i, data in tqdm(enumerate(test_loaderIND), total=len(test_loaderIND)):
        image, labels = data
        image = image.to(device)
        if type(outputsINDTest)==type(None):
            outputsINDTest=model(image).cpu().detach().numpy()
        else:
            outputsINDTest=np.concatenate((outputsINDTest, model(image).cpu().detach().numpy()), axis=0)
    
    outputsINDTestReal=None
    for i, data in tqdm(enumerate(test_loaderIND_real), total=len(test_loaderIND_real)):
        image, labels = data
        image = image.to(device)
        if type(outputsINDTestReal)==type(None):
            outputsINDTestReal=model(image).cpu().detach().numpy()
        else:
            outputsINDTestReal=np.concatenate((outputsINDTestReal, model(image).cpu().detach().numpy()), axis=0)
    
    for novelty in ['']:
        for i, data in tqdm(enumerate(test_loaderOOD[novelty] ), total=len(test_loaderOOD[novelty] )):
            image, labels = data
            image = image.to(device)
            if type(outputsOOD[novelty])==type(None):
                outputsOOD[novelty]=model(image).cpu().detach().numpy()
            else:
                outputsOOD[novelty]=np.concatenate((outputsOOD[novelty], model(image).cpu().detach().numpy()), axis=0)
    
    
    x = 0
    y = 100
    n = 600
    step = (y - x) / (n - 1)
    percentilesRange = [x + step * i for i in range(n)]
    
    #kNN
    print("Getting results for kNN")
    nbrs=NearestNeighbors(n_neighbors=30, n_jobs=-1)
    nbrs.fit(outputsINDTrain)
    
    val_scores,_ = nbrs.kneighbors(outputsINDvalidation)
    scoresTestIND,_ = nbrs.kneighbors(outputsINDTest)
    scoresTestINDReal,_ = nbrs.kneighbors(outputsINDTestReal)
    scoresTestOOD={}
    for novelty in ['']:
        scoresTestOOD[novelty],_ = nbrs.kneighbors(outputsOOD[novelty])
    print("Scores obtained")
    
    for k in [1,5,10,15,20,25,30]:
        new_val_scores = np.median(val_scores[:,:k], axis=1)
        thresholds = np.percentile(new_val_scores, percentilesRange)
        new_testIND_scores = np.median(scoresTestIND[:,:k], axis=1)
        new_testINDReal_scores = np.median(scoresTestINDReal[:,:k], axis=1)
        new_testOOD_scores={}
        for novelty in ['']:
            new_testOOD_scores[novelty] = np.median(scoresTestOOD[novelty][:,:k], axis=1)
        INDarray=np.transpose(np.array((np.reshape(new_testIND_scores, (len(new_testIND_scores), 1))>thresholds)+0, dtype=np.uint8))
        INDarrayReal=np.transpose(np.array((np.reshape(new_testINDReal_scores, (len(new_testINDReal_scores), 1))>thresholds)+0, dtype=np.uint8))
        OODarray={}
        for novelty in ['']:
            OODarray[novelty]=np.transpose(np.array((np.reshape(new_testOOD_scores[novelty], (len(new_testOOD_scores[novelty]), 1))>thresholds)+0, dtype=np.uint8))
        FPR=INDarray.sum(axis=1)/INDarray.shape[1]
        TPR=OODarray[''].sum(axis=1)/OODarray[''].shape[1]
        TPR=TPR[::-1]
        FPR=FPR[::-1]
        AUCROC=metrics.auc(FPR,TPR)
        print(0, "AUC=", AUCROC, ' k=', k)
        
        data = [args.fold, args.detectorClass, AUCROC]
        if exists('detectorsAUCkNN'+str(k)+'.csv'):
            with open('detectorsAUCkNN'+str(k)+'.csv', 'a', encoding='UTF8') as f:
                writer = csv.writer(f)
                writer.writerow(data)
        else:
            header = ['fold', 'classe', 'AUCROC']
            with open('detectorsAUCkNN'+str(k)+'.csv', 'w', encoding='UTF8') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerow(data)


        with open('results_IND_'+str(args.fold)+'_'+str(args.detectorClass)+"_kNN"+str(k)+'.results', 'wb') as handle:
            pickle.dump(np.array(INDarrayReal, dtype=np.uint8), handle, protocol=pickle.HIGHEST_PROTOCOL)
        for novelty in ['']:
            fileNameForThisNovelty='results_OOD_'+novelty+'_'+str(args.fold)+'_'+str(args.detectorClass)+"_kNN"+str(k)+'.results'
            with open(fileNameForThisNovelty, 'wb') as handle:
                pickle.dump(np.array(OODarray[novelty], dtype=np.uint8), handle, protocol=pickle.HIGHEST_PROTOCOL)

    #IF
    print("Getting results for IF")
    nbrs=IsolationForest(n_jobs=-1, bootstrap=True, n_estimators=500, random_state=42)
    nbrs.fit(outputsINDTrain)
    
    val_scores = nbrs.decision_function(outputsINDvalidation)
    scoresTestIND = nbrs.decision_function(outputsINDTest)
    scoresTestOOD={}
    for novelty in ['']:
        scoresTestOOD[novelty] = nbrs.decision_function(outputsOOD[novelty])
    scoresTestINDReal = nbrs.decision_function(outputsINDTestReal)
    
    print("Scores obtained")
    
    thresholds=np.percentile(val_scores, percentilesRange)
    INDarray=np.transpose(np.array((np.reshape(scoresTestIND, (len(scoresTestIND), 1))<thresholds)+0, dtype=np.uint8))
    INDarrayReal=np.transpose(np.array((np.reshape(scoresTestINDReal, (len(scoresTestINDReal), 1))<thresholds)+0, dtype=np.uint8))
    OODarray={}
    for novelty in ['']:
        OODarray[novelty]=np.transpose(np.array((np.reshape(scoresTestOOD[novelty], (len(scoresTestOOD[novelty]), 1))<thresholds)+0, dtype=np.uint8))

    FPR=INDarray.sum(axis=1)/INDarray.shape[1]
    TPR=OODarray[''].sum(axis=1)/OODarray[''].shape[1]
    
    TPR=TPR[::-1]
    FPR=FPR[::-1]
    AUCROC=metrics.auc(FPR,TPR)
    print(args.detectorClass, "AUC=", AUCROC)
    
    data = [args.fold, args.detectorClass, AUCROC]
    if exists('detectorsAUCIF.csv'):
        with open('detectorsAUCIF.csv', 'a', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(data)
    else:
        header = ['fold', 'classe', 'AUCROC']
        with open('detectorsAUCIF.csv', 'w', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerow(data)

    with open('results_IND_'+str(args.fold)+'_'+str(args.detectorClass)+'_IF.results', 'wb') as handle:
        pickle.dump(np.array(INDarrayReal, dtype=np.uint8), handle, protocol=pickle.HIGHEST_PROTOCOL)
    for novelty in ['']:
        fileNameForThisNovelty='results_OOD_'+novelty+'_'+str(args.fold)+'_'+str(args.detectorClass)+"_IF"+'.results'
        with open(fileNameForThisNovelty, 'wb') as handle:
            pickle.dump(np.array(OODarray[novelty], dtype=np.uint8), handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    print("Getting results for GMM")
    nbrs=GMM(n_components=75, contamination=0.1, covariance_type='diag', max_iter=500, n_init=10, reg_covar=1e-3,random_state=42)
    nbrs.fit(outputsINDTrain)
    
    val_scores = nbrs.decision_function(outputsINDvalidation)
    scoresTestIND = nbrs.decision_function(outputsINDTest)
    scoresTestOOD={}
    for novelty in ['']:
        scoresTestOOD[novelty] = nbrs.decision_function(outputsOOD[novelty])
    scoresTestINDReal = nbrs.decision_function(outputsINDTestReal)
    
    print("Scores obtained")
    
    thresholds=np.percentile(val_scores, percentilesRange)
    INDarray=np.transpose(np.array((np.reshape(scoresTestIND, (len(scoresTestIND), 1))>thresholds)+0, dtype=np.uint8))
    INDarrayReal=np.transpose(np.array((np.reshape(scoresTestINDReal, (len(scoresTestINDReal), 1))>thresholds)+0, dtype=np.uint8))
    OODarray={}
    for novelty in ['']:
        OODarray[novelty]=np.transpose(np.array((np.reshape(scoresTestOOD[novelty], (len(scoresTestOOD[novelty]), 1))>thresholds)+0, dtype=np.uint8))

    FPR=INDarray.sum(axis=1)/INDarray.shape[1]
    TPR=OODarray[''].sum(axis=1)/OODarray[''].shape[1]
    
    TPR=TPR[::-1]
    FPR=FPR[::-1]
    AUCROC=metrics.auc(FPR,TPR)
    print(args.detectorClass, "AUC=", AUCROC)
    
    data = [args.fold, args.detectorClass, AUCROC]
    if exists('detectorsAUCGMM.csv'):
        with open('detectorsAUCGMM.csv', 'a', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(data)
    else:
        header = ['fold', 'classe', 'AUCROC']
        with open('detectorsAUCGMM.csv', 'w', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerow(data)

    with open('results_IND_'+str(args.fold)+'_'+str(args.detectorClass)+'_GMM.results', 'wb') as handle:
        pickle.dump(np.array(INDarrayReal, dtype=np.uint8), handle, protocol=pickle.HIGHEST_PROTOCOL)
    for novelty in ['']:
        fileNameForThisNovelty='results_OOD_'+novelty+'_'+str(args.fold)+'_'+str(args.detectorClass)+"_GMM"+'.results'
        with open(fileNameForThisNovelty, 'wb') as handle:
            pickle.dump(np.array(OODarray[novelty], dtype=np.uint8), handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    #OCSVM
    print("Getting results for OCSVM")
    nbrs=OneClassSVM(kernel='rbf', gamma='auto', nu=0.1)
    nbrs.fit(outputsINDTrain)
    
    val_scores = nbrs.decision_function(outputsINDvalidation)
    scoresTestIND = nbrs.decision_function(outputsINDTest)
    scoresTestOOD={}
    for novelty in ['']:
        scoresTestOOD[novelty] = nbrs.decision_function(outputsOOD[novelty])
    scoresTestINDReal = nbrs.decision_function(outputsINDTestReal)
    
    print("Scores obtained")
    
    thresholds=np.percentile(val_scores, percentilesRange)
    INDarray=np.transpose(np.array((np.reshape(scoresTestIND, (len(scoresTestIND), 1))<thresholds)+0, dtype=np.uint8))
    INDarrayReal=np.transpose(np.array((np.reshape(scoresTestINDReal, (len(scoresTestINDReal), 1))<thresholds)+0, dtype=np.uint8))
    OODarray={}
    for novelty in ['']:
        OODarray[novelty]=np.transpose(np.array((np.reshape(scoresTestOOD[novelty], (len(scoresTestOOD[novelty]), 1))<thresholds)+0, dtype=np.uint8))

    FPR=INDarray.sum(axis=1)/INDarray.shape[1]
    TPR=OODarray[''].sum(axis=1)/OODarray[''].shape[1]
    
    TPR=TPR[::-1]
    FPR=FPR[::-1]
    AUCROC=metrics.auc(FPR,TPR)
    print(args.detectorClass, "AUC=", AUCROC)
    
    data = [args.fold, args.detectorClass, AUCROC]
    if exists('detectorsAUCOCSVM.csv'):
        with open('detectorsAUCOCSVM.csv', 'a', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(data)
    else:
        header = ['fold', 'classe', 'AUCROC']
        with open('detectorsAUCOCSVM.csv', 'w', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerow(data)

    with open('results_IND_'+str(args.fold)+'_'+str(args.detectorClass)+'_OCSVM.results', 'wb') as handle:
        pickle.dump(np.array(INDarrayReal, dtype=np.uint8), handle, protocol=pickle.HIGHEST_PROTOCOL)
    for novelty in ['']:
        fileNameForThisNovelty='results_OOD_'+novelty+'_'+str(args.fold)+'_'+str(args.detectorClass)+"_OCSVM"+'.results'
        with open(fileNameForThisNovelty, 'wb') as handle:
            pickle.dump(np.array(OODarray[novelty], dtype=np.uint8), handle, protocol=pickle.HIGHEST_PROTOCOL)
    
if __name__ == '__main__':
    main()
