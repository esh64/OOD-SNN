import pickle
class datasetClasse():
    def __init__(self, datasetClasse, allData=[],trainData=[], testData={}, validationData=[]):
        self.classe=str(datasetClasse)
        self.allData=allData
        self.trainData=trainData
        self.testData=testData
        self.validationData=validationData
    
    def saveDataset(self):
        pickle.dump(self, open(self.classe+'.dataset', "wb"), pickle.HIGHEST_PROTOCOL)
