import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np


LR = 0.0001
MAX_EPOCH = 50
BATCH_SIZE_1 = 64
BATCH_SIZE_2 = 1024

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def simpleFunction(x):
    val = (2 * np.sin(5 * np.pi * x)) / (5 * np.pi * x)
    return val


class FunctionApproximator1(nn.Module):
    def __init__(self):
        super(FunctionApproximator1, self).__init__()
        self.regressor = nn.Sequential(nn.Linear(1, 20),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(20, 40),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(40, 40),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(40, 20),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(20, 1))

    def forward(self, x):
        output = self.regressor(x)
        return output


X = np.random.rand(10 ** 5) #* 2 * np.pi
y = simpleFunction(X)

X_train, X_val, y_train, y_val = map(torch.tensor, train_test_split(X, y, test_size=0.2))
train_dataloader_1 = DataLoader(TensorDataset(X_train.unsqueeze(1), y_train.unsqueeze(1)), batch_size=BATCH_SIZE_1,
                              pin_memory=True, shuffle=True)
val_dataloader_1 = DataLoader(TensorDataset(X_val.unsqueeze(1), y_val.unsqueeze(1)), batch_size=BATCH_SIZE_1,
                            pin_memory=True, shuffle=True)
train_dataloader_2 = DataLoader(TensorDataset(X_train.unsqueeze(1), y_train.unsqueeze(1)), batch_size=BATCH_SIZE_2,
                              pin_memory=True, shuffle=True)
val_dataloader_2 = DataLoader(TensorDataset(X_val.unsqueeze(1), y_val.unsqueeze(1)), batch_size=BATCH_SIZE_2,
                            pin_memory=True, shuffle=True)

model1 = FunctionApproximator1().to(device)
model2 = FunctionApproximator1().to(device)
optimizer1 = optim.Adam(model1.parameters(), lr=LR)
optimizer2 = optim.Adam(model2.parameters(), lr=LR)
criterion = nn.MSELoss(reduction="mean")

# training loop for model 1
train_loss_list = []
val_loss_list = []
for epoch in range(MAX_EPOCH):
    print("epoch %d / %d" % (epoch + 1, MAX_EPOCH))
    model1.train()
    # training loop
    temp_loss_list = []
    for X_train, y_train in train_dataloader_1:
        X_train = X_train.type(torch.float32).to(device)
        y_train = y_train.type(torch.float32).to(device)
        optimizer1.zero_grad()
        score = model1(X_train)
        loss = criterion(input=score, target=y_train)
        loss.backward()
        optimizer1.step()
        temp_loss_list.append(loss.detach().cpu().numpy())
    train_loss_list.append(np.average(temp_loss_list))

    # validation
    model1.eval()
    temp_loss_list = []
    for X_val, y_val in val_dataloader_1:
        X_val = X_val.type(torch.float32).to(device)
        y_val = y_val.type(torch.float32).to(device)
        score = model1(X_val)
        loss = criterion(input=score, target=y_val)
        temp_loss_list.append(loss.detach().cpu().numpy())
    val_loss_list.append(np.average(temp_loss_list))
    print("  train loss: %.5f" % train_loss_list[-1])
    print("  val loss: %.5f" % val_loss_list[-1])


model1_y = model1(X_train)
true_y = simpleFunction(X_train)

# plot creation
fig1, (ax1, ax2) = plt.subplots(1,2)
# accuracy plot creation for model 1
ax1.scatter(X_train, model1_y.detach().numpy(), color='r',label='model1')
ax1.scatter(X_train, true_y.detach().numpy(), color ='g',label='true')
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.legend()
# loss plot creation for model 1
ax2.plot(train_loss_list, color='r', label='train')
ax2.plot(val_loss_list, color='g', label='val')
ax2.legend()
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Loss")
ax2.set_title("Training and Val Loss")

fig1.savefig("img3.3.1.png")

# training loop for model 2 (trained same way just using different models to hopefully get different result)
train_loss_list = []
val_loss_list = []
for epoch in range(MAX_EPOCH):
    print("epoch %d / %d" % (epoch + 1, MAX_EPOCH))
    model2.train()
    # training loop
    temp_loss_list = []
    for X_train, y_train in train_dataloader_2:
        X_train = X_train.type(torch.float32).to(device)
        y_train = y_train.type(torch.float32).to(device)
        optimizer2.zero_grad()
        score = model2(X_train)
        loss = criterion(input=score, target=y_train)
        loss.backward()
        optimizer2.step()
        temp_loss_list.append(loss.detach().cpu().numpy())
    train_loss_list.append(np.average(temp_loss_list))

    # validation
    model2.eval()
    temp_loss_list = []
    for X_val, y_val in val_dataloader_2:
        X_val = X_val.type(torch.float32).to(device)
        y_val = y_val.type(torch.float32).to(device)
        score = model2(X_val)
        loss = criterion(input=score, target=y_val)
        temp_loss_list.append(loss.detach().cpu().numpy())
    val_loss_list.append(np.average(temp_loss_list))

# data for plots

model2_y = model2(X_train)
true_y = simpleFunction(X_train)

# plot creation
fig2, (ax3, ax4) = plt.subplots(1,2)
# accuracy plot creation for model 2
ax3.scatter(X_train, model2_y.detach().numpy(), color='r',label='model2')
ax3.scatter(X_train, true_y.detach().numpy(), color ='g',label='true')
ax3.set_xlabel("X")
ax3.set_ylabel("Y")
ax3.legend()
# loss plot creation for model 2
ax4.plot(train_loss_list, color='r', label='train')
ax4.plot(val_loss_list, color='g', label='val')
ax4.legend()
ax4.set_xlabel("Epochs")
ax4.set_ylabel("Loss")
ax4.set_title("Training and Val Loss")

fig2.savefig("img3.3.2.png")

print(sum(param.numel() for param in model1.parameters()))
print(sum(param.numel() for param in model2.parameters()))


