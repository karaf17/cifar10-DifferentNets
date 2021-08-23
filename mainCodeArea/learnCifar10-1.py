#!/usr/bin/python
# Calling the __init__.py file imports
from libraries.libraries import *


# functions to show an image
def randomImShow(data, count):
    i = 221
    count = count % 5
    randns = np.random.randint(len(data), size=count)
    plt.figure(figsize=(5, 5))
    for a in randns:
        plt.subplot(i), plt.imshow(data[a])
        i+=1
    plt.show()

#Reformatting datas from 1024, to 32,32 with true RGB format
def reFormatDatasets(data):
    arr = []
    for a in data:
        r = a[:1024].reshape(32, 32)
        g = a[1024:2048].reshape(32, 32)
        b = a[2048:].reshape(32, 32)
        arr.append(np.dstack((r, g, b)))
    arr = np.array(arr)

    return arr

# To manipulate and label dataset
class ManipulateData:
    def __init__(self, path, trainSubPath, testSubPath):
        self.path = path
        self.trainSubPath = trainSubPath
        self.testSubPath = testSubPath
        self.trainDataFile = None
        self.testDataFile = None
        self.trainData = None
        self.trainLabels = None
        self.testData = None
        self.testLabels = None
        self.trainTry = None
        self.callFunctionsInOrder()

    def callFunctionsInOrder(self):
        self.createTrainTestSet()
        self.trainData = reFormatDatasets(self.trainData)
        self.testData = reFormatDatasets(self.testData)
        self.loadTheData()


    # unpickle from dataset
    def unPickle(self, filename):
        with open(filename, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    # Create the array datasets of train and test with labels
    def uploadData(self):
        self.trainDataFile = os.path.join(self.path, "", self.trainSubPath)
        self.testDataFile = os.path.join(self.path, "", self.testSubPath)

    def createTrainTestSet(self):
        self.uploadData()
        unpackTrain = self.unPickle(self.trainDataFile)
        unpackTest = self.unPickle(self.testDataFile)
        namelist = list(unpackTrain)
        print(namelist)
        self.trainLabels = np.array(unpackTrain[namelist[1]])
        self.trainData = unpackTrain[namelist[2]]/255
        self.trainTry = DataLoader(unpackTrain , batch_size=10, shuffle=True)
        self.testLabels = np.array(unpackTest[namelist[1]])
        self.testData = unpackTest[namelist[2]]/255

    def loadTheData(self):
        self.trainData  = DataLoader(self.trainData , batch_size=10, shuffle=True)
        self.testData  = DataLoader(self.testData , batch_size=10, shuffle=True)

# Create the main class for self defined network
class Net(Module):
    def __init__(self):
        super(Net, self).__init__()
        self.network = None
        self.layers = None
        self.linear_layers = None
        self.fillTheLayers()

    def fillTheLayers(self):
        self.layers = Sequential(Conv2d(3, 2, kernel_size=(3,3), stride=1, padding=(1,1), padding_mode = 'replicate'),
                                 MaxPool2d(2,2),
                                 Conv2d(2, 1, kernel_size=(3,3), stride=1, padding=(1,1), padding_mode = 'replicate'),
                                 MaxPool2d(2,2)
                                 )
        self.linear_layers = Sequential(Linear(64,32),
                                        Linear(32,10))

    # Defining the forward pass
    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

# Give input as the data paths here
def getDataset():
    path = "/home/tbtk-centos-net2/Development/fkara/pycharm_projects/cifar-10/datasetArea/cifar-10-batches-py"
    trainSubPath = "data_batch_5"
    testSubPath = "test_batch"
    md = ManipulateData(path=path,
                        trainSubPath=trainSubPath,
                        testSubPath=testSubPath)
    return md.trainData, md.trainLabels, md.trainTry

# Train test splitting module
def splitTrainSet(trainSet, labelSet):
    trainX, valX, trainY, valY = train_test_split(trainSet, labelSet, test_size=0.1, random_state=True)
    trainX = trainX.reshape(9000, 3, 32, 32)
    valX = valX.reshape(1000, 3, 32, 32)
    trainY = trainY.astype(int)
    valY = valY.astype(int)
    trainX = torch.from_numpy(trainX)
    trainY = torch.from_numpy(trainY)
    valX = torch.from_numpy(valX)
    valY = torch.from_numpy(valY)

    return trainX, trainY, valX, valY

def createModel():
    # defining the model
    model = Net()
    # defining the optimizer
    optimizer = Adam(model.parameters(), lr=0.07)
    # defining the loss function
    criterion = CrossEntropyLoss()
    # checking if GPU is available
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    print(model)
    return model, optimizer, criterion

def train_val_dataset(dataset, val_split=0.25):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets

# Main Training Module
def trainTheModel(epochNum):
    train_losses = []
    val_losses = []
    # Get the dataset and convert it into train and val sets
    train, test, tryTrain = getDataset()
    print(type(tryTrain))

    trainX, trainY, valX, valY = splitTrainSet(train, test)
    trainX = trainX.float()
    valX = valX.float()
    # Converting the torch formatted arrays to Variable arrays
    xTrain, yTrain = Variable(trainX), Variable(trainY)
    xVal, yVal = Variable(valX), Variable(valY)

    if torch.cuda.is_available():
        xTrain = xTrain.cuda()
        yTrain = yTrain.cuda()
        xVal = xVal.cuda()
        yVal = yVal.cuda()
    print(xTrain[0], yTrain[0])
    # Create the model
    model, optimizer, criterion = createModel()


    for epoch in range(epochNum):
        model.train()
        # clearing the Gradients of the model parameters
        optimizer.zero_grad()
        model.eval()
        # prediction for training and validation set
        output_train = model(xTrain)
        output_val = model(xVal)

        # computing the training and validation loss
        loss_train = criterion(output_train, yTrain)
        loss_val = criterion(output_val, yVal)

        train_losses.append(loss_train)
        val_losses.append(loss_val)

        # computing the updated weights of all the model parameters
        loss_train.backward()
        optimizer.step()
        tr_loss = loss_train.item()
        print('Epoch : ',epoch+1)
        print('val loss :', loss_val)
        print('train loss:', loss_train)

    # Plot the train and validation losses after training
    plt.figure()
    plt.plot(val_losses, color='magenta')
    plt.title("Validation Losses")
    plt.xlabel("Step number")
    plt.ylabel("Val_Loss")
    plt.figure()
    plt.plot(train_losses)
    plt.title("Train Losses")
    plt.xlabel("Step number")
    plt.ylabel("Train_Loss")
    plt.show()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    td,ted,tt = getDataset()
    print(type(ted))
    # print("main")
    # tr, ts = getDataset()
    # randomImShow(tr, 4)

