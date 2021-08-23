from libraries.libraries import *
from utils import *
from dataLoader import *
BATCH_SIZE = 4

def doTraining(epochNum):
    model, opt, crt = createModel()
    device = torch.device('cuda:0')
    cd = CustomDataset(doTransform=False)
    cd.loadAlreadyDataset(TrainLoad=True)
    data = cd.aTrain
    train(model, opt, crt, device, epochNum, data, BATCH_SIZE)
    return model


def doTesting(model):
    # model, _, _ = createModel()
    device = torch.device('cuda:0')
    cd = CustomDataset(doTransform=False)
    cd.loadAlreadyDataset(TrainLoad=False)
    data = cd.aTest
    test(model, data, device, BATCH_SIZE)

def createModel():
    # defining the model
    model = AlexNet()
    # defining the optimizer (ADAM Generally gives more precise result than SGD)
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    # optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
    # defining the loss function
    criterion = CrossEntropyLoss()
    # checking if GPU is available
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    print(model)
    return model, optimizer, criterion


class AlexNet(Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = Conv2d(3, 64, 11, stride=4, padding=2)
        self.conv2 = Conv2d(64, 192, 5, stride=1, padding=2)
        self.conv3 = Conv2d(192, 384, 3, stride=1, padding=1)
        self.conv4 = Conv2d(384, 256, 3, stride=1, padding=1)
        self.conv5 = Conv2d(256, 256, 3, stride=1, padding=1)
        self.fc1 = Linear(9216, 4096)
        self.fc2 = Linear(4096, 1024)
        self.fc3 = Linear(1024, 10)
        self.avgPool = AdaptiveAvgPool2d((6, 6))
        self.pool = MaxPool2d(3, 2)

    def forward(self, x):
        x = relu(self.conv1(x))
        x = self.pool(x)
        x = relu(self.conv2(x))
        x = self.pool(x)
        x = relu(self.conv3(x))
        x = relu(self.conv4(x))
        x = self.pool(relu(self.conv5(x)))
        x = self.avgPool(x)
        x = dropout(flatten(x, 1), p=0.5)
        x = dropout(relu(self.fc1(x)), p=0.5)
        x = relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    model = doTraining(10)
    doTesting(model)
