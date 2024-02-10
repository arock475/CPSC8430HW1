import torch
from torch import  nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim

LEARNING_RATE = 0.0001
MOMENTUM = 0.9
MAX_EPOCHS = 10
BATCH_SIZE = 4
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
class ImageClass1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 100)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = nn.functional.relu(self.fc1(x))
        return x


if __name__ == '__main__':
    # create transform to normalize data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # download and normalize data
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    #label randomization
    print(trainset.targets[0])
    #trainset.targets = np.random.permutation(trainset.targets)
    print(trainset.targets[0])
    model1 = ImageClass1()
    criterion = nn.CrossEntropyLoss()
    optimizer1 = optim.SGD(model1.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    # train model 1
    loss_list_1 = []
    acc_list_1 = []
    test_loss_list = []
    for epoch in range(MAX_EPOCHS):
        running_loss = 0.0
        loss_temp = []
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            labels = torch.randint(0, 10, labels.size())
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
        test_loss = 0.0
        temp_loss_list = []
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                # calculate outputs by running images through the network
                outputs = model1(images)
                # the class with the highest energy is what we choose as prediction
                loss = criterion(outputs, labels)
                temp_loss_list.append(loss.detach().cpu().numpy())
            test_loss_list.append(np.average(temp_loss_list))
        print(test_loss_list[-1])
    print('Finished Training Model 1')


    plt.plot(loss_list_1, color='r', label='train_loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(test_loss_list, color='g', label='test_loss')
    plt.legend()

    plt.savefig("img3.1.png")