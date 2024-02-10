import torch
from torch import  nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim

LEARNING_RATE = 0.0001
MOMENTUM = 0.9
MAX_EPOCHS = 25
BATCH_SIZE = 4
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
class ImageClass1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 10)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = nn.functional.relu(self.fc1(x))
        return x

class ImageClass2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        return x


if __name__ == '__main__':
    # create transform to normalize data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # download and normalize data
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    model1 = ImageClass1()
    model2 = ImageClass2()
    criterion = nn.CrossEntropyLoss()
    optimizer1 = optim.SGD(model1.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    optimizer2 = optim.SGD(model2.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    # train model 1
    loss_list_1 = []
    acc_list_1 = []
    for epoch in range(MAX_EPOCHS):
        running_loss = 0.0
        loss_temp = []
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer1.zero_grad()
            outputs = model1(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer1.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                loss_temp.append(running_loss / 2000)
                running_loss = 0.0
        # get loss for epoch
        loss_final = 0.0
        for i in loss_temp:
            loss_final += i
        loss_final = loss_final/ len(loss_temp)
        loss_list_1.append(loss_final)
        print("Loss = " + str(loss_final))
        # test accuracy on test data
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                # calculate outputs by running images through the network
                outputs = model1(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc_list_1.append(correct / total)
        print(f'Accuracy: {correct / total}')
    print('Finished Training Model 1')
    # train model 2
    loss_list_2 = []
    acc_list_2 = []
    for epoch in range(MAX_EPOCHS):
        running_loss = 0.0
        loss_temp = []
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer2.zero_grad()
            outputs = model2(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer2.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                loss_temp.append(running_loss / 2000)
                running_loss = 0.0
        # get loss for epoch
        loss_final = 0.0
        for i in loss_temp:
            loss_final += i
        loss_final = loss_final / len(loss_temp)
        loss_list_2.append(loss_final)
        print("Loss = " + str(loss_final))
        # test accuracy on test data
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                # calculate outputs by running images through the network
                outputs = model2(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc_list_2.append(correct / total)
        print(f'Accuracy: {correct / total}')
    print('Finished Training Model 2')

    fig1, (ax1, ax2) = plt.subplots(1,2)
    # accuracy plot creation
    ax1.plot(acc_list_1, color='r',label='model 1')
    ax1.plot(acc_list_2, color ='g',label='model 2')
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    # loss plot creation
    ax2.plot(loss_list_1, color='r', label='model 1')
    ax2.plot(loss_list_2, color='g', label='model 2')
    ax2.legend()
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Loss")
    ax2.set_title("Training and Val Loss")

    fig1.savefig("img1.2.png")