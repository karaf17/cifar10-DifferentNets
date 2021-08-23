import numpy as np

from libraries.libraries import *

def reFormatDatasets(data, shape=32):
    fShape = shape*shape
    sShape = shape*shape*2
    r = data[:fShape].reshape(shape, shape)
    g = data[fShape:sShape].reshape(shape, shape)
    b = data[sShape:].reshape(shape, shape)
    arr = np.array([r, g, b])
    arr = np.moveaxis(arr, 0, 2)
    return arr


def train(model, optimizer, criterion, device, epoch, data, batch_size=10):
    model.train()
    dataLoader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=2)
    dataProvider = tqdm(dataLoader, smoothing=0, mininterval=1.0)
    for epochs in range(epoch):
        running_loss = 0.0
        totalLossList = []
        total = 0
        correct = 0
        for i, (fields, target) in enumerate(dataProvider):
            fields, target = fields.to(device, dtype=torch.float32), target.to(device, dtype=torch.int64)
            optimizer.zero_grad()
            # predict
            y = model(fields)
            del fields
            torch.cuda.empty_cache()
            # Do self prediction on model:
            _, predicted = torch.max(y.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            # calculate loss for predictions
            loss = criterion(y, target)
            # backpropagation for weight adjusting
            del target
            torch.cuda.empty_cache()
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            totalLossList.append(running_loss)
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (epochs + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

        if (epochs + 1) % 3 == 0 and optimizer.param_groups[0]['lr'] > 1e-4:
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / 1.1
        print('Self accuracy of the network on the images: %', (100 * correct / total))
        stdVal = np.std(totalLossList)
        meanVal = np.mean(totalLossList)
        print("Values of Process--> Mean: ", meanVal, "STD: ", stdVal)
        print("The New Learning Rate is: ", optimizer.param_groups[0]['lr'])
        print("whole epoch: ", epochs + 1, "Training Loss: ", running_loss)


def test(model, data, device, batch_size=10):
    testLoader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=2)
    predictionArray = []
    correct = 0
    total = 0
    with torch.no_grad():
        for vals in testLoader:
            images, labels = vals
            images, labels = images.to(device, dtype=torch.float32), labels.to(device, dtype=torch.int64)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            predictionArray.append(predicted)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %', (100 * correct / total))


def doTransforms(data):
    # data = data.astype(np.int8)
    trnfs = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
                                transforms.RandomCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                ])
    # print(trnfs(data).shape)
    # return data
    return trnfs(data)
