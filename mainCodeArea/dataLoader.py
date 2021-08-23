from utils import *
from libraries.libraries import *


class CustomDataset:
    def __init__(self, doTransform=False):
        self.path = "/home/tbtk-centos-net2/Development/fkara/pycharm_projects/cifar10-new/datasetArea/cifar-10-batches-py"
        self.trainSubPath = "data_batch_"
        self.testSubPath = "test_batch"
        self.doTransform = doTransform
        self.tDataFile = None
        self.data = []
        self.labels = []
        self.tData = []
        self.aTest = None
        self.aTrain = None

    def loadDataset(self, trainBatch=[], trainLoad=True):
        if len(trainBatch) == 0:
            trainBatch = [1]
        if trainLoad:
            print("reading train data ID: ", trainBatch)
            for element in trainBatch:
                self.tDataFile = os.path.join(self.path, "", self.trainSubPath + str(element))
                with open(self.tDataFile, 'rb') as file:
                    dict1 = pickle.load(file, encoding='bytes')
                namelist = list(dict1)
                self.labels = dict1[namelist[1]]
                self.tData = dict1[namelist[2]]
                for i in range(len(self.tData)):
                    targets = reFormatDatasets(self.tData[i])
                    with torch.no_grad():
                        targets = doTransforms(targets)
                    features = self.labels[i]
                    self.data.append([targets, features])
                    del targets
                    torch.cuda.empty_cache()

        if not trainLoad:
            print("reading test data")
            self.tDataFile = os.path.join(self.path, "", self.testSubPath)
            with open(self.tDataFile, 'rb') as file:
                dict1 = pickle.load(file, encoding='bytes')
            namelist = list(dict1)
            self.labels = dict1[namelist[1]]
            self.tData = dict1[namelist[2]]
            for i in range(len(self.tData)):
                targets = reFormatDatasets(self.tData[i])
                with torch.no_grad():
                    targets = doTransforms(targets)
                features = self.labels[i]
                self.data.append([targets, features])
                del targets
                torch.cuda.empty_cache()

    def loadAlreadyDataset(self, TrainLoad=True):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        if TrainLoad:
            self.aTrain = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        else:
            self.aTest = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
