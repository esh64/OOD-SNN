import Library
import numpy as np
import argparse
from tqdm.auto import tqdm
import csv
from os.path import exists
import pickle

from sklearn import metrics

parser = argparse.ArgumentParser(description='Tensorflow FMNIST NoveltyDetection')
parser.add_argument('--fold', type=int, default=0, metavar='N',
                    help='input fold number (default: 0)')

args = parser.parse_args()

for novelty in ['']:
    for method in ['kNN1', 'kNN5', 'kNN10', 'kNN15', 'kNN20', 'kNN25', 'kNN30', 'IF', 'OCSVM', 'GMM']:
        setted=False
        for name in range(3):
            loadedFile=open('results_IND_'+str(args.fold)+'_'+str(name)+"_"+method+'.results', 'rb')
            scoresIND=pickle.load(loadedFile)
            loadedFile=open('results_OOD_'+novelty+'_'+str(args.fold)+'_'+str(name)+"_"+method+'.results', 'rb')
            scoresOOD=pickle.load(loadedFile)
            if not setted:
                IND_size=scoresIND.shape[1]
                OOD_size=scoresOOD.shape[1]
                somaIND=np.zeros((600,IND_size))
                somaOOD=np.zeros((600,OOD_size))
                setted=True
            
            somaIND+=scoresIND
            somaOOD+=scoresOOD
            
        TPR=np.sum((somaOOD==3)+0, axis=1)/OOD_size
        FPR=np.sum((somaIND==3)+0, axis=1)/IND_size
        
        div=np.sum((((somaIND==3)+0)), axis=1)+np.sum((((somaOOD==3)+0)), axis=1)
        indexes=np.where(div==0)
        np.put(div, indexes[0], 1)
        quo=np.sum((((somaOOD==3)+0)), axis=1)
        np.put(quo, indexes[0], 1)
        PPV=quo/div

        
        if 'kNN' in method or 'GMM'==method:
            PPV=PPV[::-1]
            TPR=TPR[::-1]
            FPR=FPR[::-1]    
        
        for fp, tp in zip(FPR, TPR):
            if tp >= 0.95:
                fpr95tpr=fp
                break
            fpr95tpr=None
        
        for fp, tp in zip(FPR, TPR):
            if tp >= 0.8:
                fpr80tpr=fp
                break
            fpr80tpr=None
        
        data = [args.fold, metrics.auc(FPR, TPR), metrics.auc(TPR, PPV),fpr95tpr, fpr80tpr]
        if exists('SNN'+method+novelty+'.csv'):
            with open('SNN'+method+novelty+'.csv', 'a', encoding='UTF8') as f:
                writer = csv.writer(f)
                writer.writerow(data)
        else:
            header = ['fold', 'AUCROC', 'AUPRC','FPR95TPR', 'FPR80TPR']
            with open('SNN'+method+novelty+'.csv', 'w', encoding='UTF8') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerow(data) 
