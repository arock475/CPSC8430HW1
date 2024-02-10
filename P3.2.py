import torch
from torch import  nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim

LEARNING_RATE = 0.0001
MOMENTUM = 0.9
MAX_EPOCHS = 2
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
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 10)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = nn.functional.relu(self.fc1(x))
        return x
class ImageClass3(nn.Module):
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
class ImageClass4(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x
class ImageClass5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 46, 5)
        self.fc1 = nn.Linear(46 * 5 * 5, 150)
        self.fc2 = nn.Linear(150, 104)
        self.fc3 = nn.Linear(104, 10)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x
class ImageClass6(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 46, 5)
        self.fc1 = nn.Linear(46 * 5 * 5, 150)
        self.fc2 = nn.Linear(150, 154)
        self.fc3 = nn.Linear(154, 104)
        self.fc4 = nn.Linear(104, 10)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.fc4(x)
        return x
class ImageClass7(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 46, 5)
        self.fc1 = nn.Linear(46 * 5 * 5, 250)
        self.fc2 = nn.Linear(250, 154)
        self.fc3 = nn.Linear(154, 104)
        self.fc4 = nn.Linear(104, 10)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.fc4(x)
        return x
class ImageClass8(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 46, 5)
        self.fc1 = nn.Linear(46 * 5 * 5, 250)
        self.fc2 = nn.Linear(250, 250)
        self.fc3 = nn.Linear(250, 150)
        self.fc4 = nn.Linear(150, 10)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.fc4(x)
        return x
class ImageClass9(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 46, 5)
        self.fc1 = nn.Linear(46 * 5 * 5, 250)
        self.fc2 = nn.Linear(250, 250)
        self.fc3 = nn.Linear(250, 150)
        self.fc4 = nn.Linear(150, 85)
        self.fc5 = nn.Linear(85, 10)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        return x
class ImageClass10(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 166, 5)
        self.fc1 = nn.Linear(166 * 5 * 5, 350)
        self.fc2 = nn.Linear(350, 450)
        self.fc3 = nn.Linear(450, 250)
        self.fc4 = nn.Linear(250, 85)
        self.fc5 = nn.Linear(85, 10)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        return x

def train(model_num, model, optimizer):
    training_loss = []
    test_loss = []
    training_acc = []
    test_acc = []
    for epoch in range(MAX_EPOCHS):
        training_loss_temp = []
        test_loss_temp = []
        correct = 0.0
        total = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            training_loss_temp.append(loss.detach().cpu().numpy())
        # get training loss
        training_loss.append(np.average(training_loss_temp))
        print("Loss = " + str(training_loss[-1]))
        # get training accuracy
        accuracy = correct / len(trainset)
        training_acc.append(accuracy)
        print("Training Acc = " + str(training_acc[-1]))
        # test accuracy on test data
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                loss = criterion(input=outputs, target=labels)
                test_loss_temp.append(loss.detach().numpy())
        test_acc.append(correct / total)
        print("Test Accuracy = " + str(test_acc[-1]))
        test_loss.append(np.average(test_loss_temp))
        print("Test Loss = " + str(test_loss[-1]))
    print('Finished Training Model' + str(model_num))
    return training_loss, training_acc, test_loss, test_acc


if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    models = [ImageClass1(), ImageClass2(), ImageClass3(), ImageClass4(), ImageClass5(), ImageClass6(), ImageClass7(),
              ImageClass8(), ImageClass9(), ImageClass10()]

    criterion = nn.CrossEntropyLoss()
    optimizers = [optim.SGD(models[0].parameters(), lr=LEARNING_RATE, momentum=MOMENTUM),
                  optim.SGD(models[1].parameters(), lr=LEARNING_RATE, momentum=MOMENTUM),
                  optim.SGD(models[2].parameters(), lr=LEARNING_RATE, momentum=MOMENTUM),
                  optim.SGD(models[3].parameters(), lr=LEARNING_RATE, momentum=MOMENTUM),
                  optim.SGD(models[4].parameters(), lr=LEARNING_RATE, momentum=MOMENTUM),
                  optim.SGD(models[5].parameters(), lr=LEARNING_RATE, momentum=MOMENTUM),
                  optim.SGD(models[6].parameters(), lr=LEARNING_RATE, momentum=MOMENTUM),
                  optim.SGD(models[7].parameters(), lr=LEARNING_RATE, momentum=MOMENTUM),
                  optim.SGD(models[8].parameters(), lr=LEARNING_RATE, momentum=MOMENTUM),
                  optim.SGD(models[9].parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)]

    fig1, (ax1, ax2) = plt.subplots(1, 2)
    num_params_list = []
    training_acc_list_list = []
    training_loss_list_list = []
    test_acc_list_list = []
    test_loss_list_list = []
    for model_num in range(10):
        training_loss_list, training_acc_list, test_loss_list, test_acc_list = train(model_num, models[model_num], optimizers[model_num])
        num_params = sum(param.numel() for param in models[model_num].parameters())
        print("Number of Parameters" + str(num_params))
        num_params_list.append(num_params)
        training_acc_list_list.append(training_acc_list[-1])
        training_loss_list_list.append(training_loss_list[-1])
        test_acc_list_list.append(test_acc_list[-1])
        test_loss_list_list.append(test_loss_list[-1])
    ax1.scatter(num_params_list, training_acc_list_list, color='r', label='training accuracy')
    ax1.scatter(num_params_list, test_acc_list_list, color='g', label='test accuracy')
    ax2.scatter(num_params_list, training_loss_list_list, color='r', label='training loss')
    ax2.scatter(num_params_list, test_loss_list_list, color='g', label='test loss')
    ax1.set_xlabel("Parameters")
    ax1.set_ylabel("Accuracy")
    ax2.set_xlabel("Parameters")
    ax2.set_ylabel("Loss")
    ax1.legend()
    ax2.legend()
    fig1.savefig("img3.2.png")