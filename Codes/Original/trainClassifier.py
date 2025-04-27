from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import Library
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(
        self, best_valid_loss=float(0)
    ):
        self.best_valid_loss = best_valid_loss
        
    def __call__(
        self, current_valid_loss, 
        epoch, model, optimizer, fold
    ):
        #if current_valid_loss1 >= self.best_valid_loss and current_valid_loss2 <= self.best_valid_loss2:
        if current_valid_loss > self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': None,
                }, 'fmnist_lenet_'+str(fold)+'.pth')

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc = nn.Linear(400, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, 3)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out
    
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    return correct / len(test_loader.dataset)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Genomics Example')
    parser.add_argument('--fold', type=int, default=0, metavar='N',
                        help='input fold number (default: 0)')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
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

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        kwargs.update({'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True},
                     )
        
    trainDataX=None
    trainDataY=[]
    validationDataX=None
    validationDataY=[]
    testDataX=None
    testDataY=[]

    for name in range(3):
        classDataset=Library.importDataset(str(name))
        if type(trainDataX)==type(None):
            trainDataX=classDataset.trainData[args.fold]
            trainDataY+=[name]*len(classDataset.trainData[args.fold])
            validationDataX=classDataset.validationData[args.fold]
            validationDataY+=[name]*len(classDataset.validationData[args.fold])
        else:
            trainDataX=np.concatenate((trainDataX, classDataset.trainData[args.fold]), axis=0)
            trainDataY+=[name]*len(classDataset.trainData[args.fold])
            validationDataX=np.concatenate((validationDataX, classDataset.validationData[args.fold]), axis=0)
            validationDataY+=[name]*len(classDataset.validationData[args.fold])
          
    if trainDataX.ndim == 3:
        trainDataX = np.expand_dims(trainDataX, axis=-1)
        validationDataX = np.expand_dims(validationDataX, axis=-1)

    trainDataX=np.moveaxis(trainDataX, -1, 1)
    validationDataX=np.moveaxis(validationDataX, -1, 1)
    trainDataY=np.array(trainDataY)
    validationDataY=np.array(validationDataY)
    tensorTrainDataX = torch.from_numpy(trainDataX).float() / 255.0
    tensorTrainDataY = torch.from_numpy(trainDataY).long()
    tensorValidationDataX = torch.from_numpy(validationDataX).float() / 255.0
    tensorValidationDataY = torch.from_numpy(validationDataY).long()
    dataset1=TensorDataset(tensorTrainDataX, tensorTrainDataY)
    dataset2=TensorDataset(tensorValidationDataX, tensorValidationDataY)
    train_loader = torch.utils.data.DataLoader(dataset1,**kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **kwargs)
    
    print("Datasets prontos")
    save_best_model = SaveBestModel()
    
    model = LeNet().to(device)
    optimizer = optim.Adam(model.parameters())
    #criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    #scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    
    for epoch in range(1, args.epochs + 1):
        print("Epoch:" +str(epoch))
        train(args, model, device, train_loader, optimizer, epoch)
        epoch_acc=test(model, device, test_loader)
        scheduler.step(epoch_acc)
        #scheduler.step()
        save_best_model(epoch_acc, epoch, model, optimizer, args.fold)
        print('-'*10)
    
    #save_model(args.epochs, model, optimizer, args.fold)
    print('TRAINING COMPLETE')

if __name__ == '__main__':
    main()
