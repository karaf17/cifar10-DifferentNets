from libraries.libraries import *
from utils import *
from dataLoader import *
BATCH_SIZE = 20

def doTraining(epochNum):
    model, opt, crt = createModel()
    device = torch.device('cuda:0')
    cd = CustomDataset(doTransform=False)
    cd.loadDataset(trainBatch=[1, 2, 3, 4, 5], trainLoad=True)
    data = cd.data
    train(model, opt, crt, device, epochNum, data, BATCH_SIZE)
    return model


def doTesting(model):
    # model, _, _ = createModel()
    device = torch.device('cuda:0')
    cd = CustomDataset(doTransform=False)
    cd.loadDataset(trainLoad=False)
    data = cd.data
    test(model, data, device, BATCH_SIZE)

def createModel():
    # defining the model
    model = AlexNet()
    # defining the optimizer (ADAM Generally gives more precise result than SGD)
    # optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
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
        self.conv1 = Conv2d(3, 96, 11, stride=4, padding=1)
        self.conv2 = Conv2d(96, 256, 5, stride=1, padding=2, groups=2)
        self.conv3 = Conv2d(256, 384, 3, stride=1, padding=1)
        self.conv4 = Conv2d(384, 384, 3, stride=1, padding=1, groups=2)
        self.conv5 = Conv2d(384, 256, 3, stride=1, padding=1, groups=2)
        self.fc1 = Linear(256 * 13 * 13, 256 * 13)
        self.fc2 = Linear(256 * 13, 64)
        self.fc3 = Linear(64, 10)

    def forward(self, x):
        x = relu(self.conv1(x))
        x = max_pool2d(x, 2)
        x = relu(self.conv2(x))
        x = max_pool2d(x, 2)
        x = relu(self.conv3(x))
        x = relu(self.conv4(x))
        x = relu(self.conv5(x))
        x = flatten(x, 1)
        x = dropout(relu(self.fc1(x)), p=0.2)
        x = dropout(relu(self.fc2(x)), p=0.2)
        x = self.fc3(x)

        return x


if __name__ == '__main__':
    model = doTraining(20)
    doTesting(model)
