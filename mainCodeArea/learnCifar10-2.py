from libraries.libraries import *
from utils import *
from dataLoader import CustomDataset
BATCH_SIZE = 256


def createModel():
    # defining the model
    model = Net()
    # defining the optimizer
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    # optimizer = Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    # defining the loss function
    criterion = CrossEntropyLoss()
    # checking if GPU is available
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    print(model)
    print("First Learning Rate: ", optimizer.param_groups[0]['lr'])
    return model, optimizer, criterion


# Create the main class for self defined network
class Net(Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(3, 9, kernel_size=(9, 9), stride=4, padding=2, padding_mode='reflect')
        self.pool = MaxPool2d(3, 2)
        self.conv2 = Conv2d(9, 16, kernel_size=(7, 7),  padding=2, padding_mode='circular')
        self.conv3 = Conv2d(16, 27, kernel_size=(3, 3), padding=2, padding_mode='replicate')
        self.conv4 = Conv2d(27, 36, kernel_size=(3, 3), padding=1, padding_mode='zeros')
        self.conv5 = Conv2d(36, 36, kernel_size=(3, 3), padding=1, padding_mode='zeros')
        self.fc1 = Linear(36 * 14 * 14, 36 * 14)
        self.fc2 = Linear(36 * 14, 36)
        self.fc3 = Linear(36, 10)

    # Defining the forward pass
    def forward(self, x):
        x = self.pool(rrelu(self.conv1(x)))
        x = self.pool(rrelu(self.conv2(x)))
        x = rrelu(self.conv3(x))
        x = rrelu(self.conv4(x))
        x = rrelu(self.conv5(x))
        x = dropout(flatten(x, 1))
        x = rrelu(self.fc1(x))
        BatchNorm1d(36*14*14)
        x = celu(self.fc2(x), alpha=2)
        BatchNorm1d(36*14)
        x = celu(self.fc3(x), alpha=2)
        return x


def doTraining(epochNum, model, opt, crt):
    device = torch.device('cuda:0')
    model.to(device)
    cd = CustomDataset()
    cd.loadDataset(trainBatch=[1], trainLoad=True)
    data = cd.data
    # cd.loadAlreadyDataset(TrainLoad=True)
    # data = cd.aTrain
    train(model, opt, crt, device, epochNum, data, BATCH_SIZE)


def doTesting(model):
    # model, _, _ = createModel()
    device = torch.device('cuda:0')
    cd = CustomDataset()
    cd.loadDataset(trainLoad=False)
    data = cd.data
    # cd.loadAlreadyDataset(TrainLoad=False)
    # data = cd.aTest
    test(model, data, device, 1000)


if __name__ == '__main__':
    model, opt, crt = createModel()
    for _ in range(4):
        doTraining(5, model, opt, crt)
        doTesting(model)


