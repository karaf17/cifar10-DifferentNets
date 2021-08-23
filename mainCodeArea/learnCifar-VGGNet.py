from libraries.libraries import *
from utils import *
from dataLoader import CustomDataset
BATCH_SIZE = 10

def createModel():
    # defining the model
    model = VGGNet()
    # defining the optimizer
    # optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = Adam(model.parameters(), lr=0.007)  # , weight_decay=1e-12)
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
class VGGNet(Module):
    def __init__(self):
        super(VGGNet, self).__init__()
        # first blocks
        self.conv1 = Conv2d(3, 64, kernel_size=(3, 3), padding_mode='reflect')
        self.conv2 = Conv2d(64, 64, kernel_size=(3, 3), padding_mode='circular')
        # second blocks
        self.conv3 = Conv2d(64, 128, kernel_size=(3, 3), padding_mode='zeros')
        self.conv4 = Conv2d(128, 128, kernel_size=(3, 3), padding_mode='zeros')
        # third blocks
        self.conv5 = Conv2d(128, 256, kernel_size=(3, 3), padding_mode='replicate')
        self.conv6 = Conv2d(256, 256, kernel_size=(3, 3), padding_mode='replicate')
        self.conv7 = Conv2d(256, 256, kernel_size=(1, 1), padding_mode='replicate')
        # fourth blocks
        self.conv8 = Conv2d(256, 512, kernel_size=(3, 3), padding_mode='replicate')
        self.conv9 = Conv2d(512, 512, kernel_size=(3, 3), padding_mode='replicate')
        self.conv10 = Conv2d(512, 512, kernel_size=(1, 1), padding_mode='replicate')
        # fifth blocks
        self.conv11 = Conv2d(512, 512, kernel_size=(3, 3), padding_mode='replicate')
        self.conv12 = Conv2d(512, 512, kernel_size=(3, 3), padding_mode='replicate')
        self.conv13 = Conv2d(512, 512, kernel_size=(1, 1), padding_mode='replicate')
        # fully connected layers
        self.fc1 = Linear(512*3*3, 512*3)
        self.fc2 = Linear(512*3, 512)
        self.fc3 = Linear(512, 10)
        # main pooling layer
        self.pool = MaxPool2d(2, 2)

    # Defining the forward pass
    def forward(self, x):
        x = relu(self.conv1(x))
        x = self.pool(relu(self.conv2(x)))
        x = relu(self.conv3(x))
        x = self.pool(relu(self.conv4(x)))
        x = relu(self.conv5(x))
        x = relu(self.conv6(x))
        x = self.pool(relu(self.conv7(x)))
        x = relu(self.conv8(x))
        x = relu(self.conv9(x))
        x = self.pool(relu(self.conv10(x)))
        x = relu(self.conv11(x))
        x = relu(self.conv12(x))
        x = self.pool(relu(self.conv13(x)))
        x = flatten(x, 1)
        x = dropout(self.fc1(x), p=0.2)
        x = dropout(self.fc2(x), p=0.2)
        x = softmax(self.fc3(x), dim=1)
        return x


def doTraining(epochNum):
    model, opt, crt = createModel()
    device = torch.device('cuda:0')
    model.to(device)
    cd = CustomDataset()
    # cd.loadDataset(trainBatch=[1], trainLoad=True)
    # data = cd.data
    cd.loadAlreadyDataset(TrainLoad=True)
    data = cd.aTrain
    train(model, opt, crt, device, epochNum, data, BATCH_SIZE)
    return model


def doTesting(model):
    # model, _, _ = createModel()
    device = torch.device('cuda:0')
    cd = CustomDataset()
    # cd.loadDataset(trainLoad=False)
    # data = cd.data
    cd.loadAlreadyDataset(TrainLoad=False)
    data = cd.aTest
    test(model, data, device, BATCH_SIZE)


if __name__ == '__main__':
    model = doTraining(10)
    doTesting(model)
