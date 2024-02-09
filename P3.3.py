import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np


LR_1 = 0.0001
LR_2 = 0.001
MAX_EPOCH = 10
BATCH_SIZE = 64

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

def interpolate_models(model_1, model_2, alpha):
    new_model = FunctionApproximator1()
    for (name1, param1), (name2, param2) in zip(model_1.named_parameters(), model_2.named_parameters()):
        new_param = alpha * param1.data + (1 - alpha) * param2.data
        setattr(new_model, 'param', nn.Parameter(new_param))
    return new_model

X = np.random.rand(10 ** 5) #* 2 * np.pi
y = simpleFunction(X)

X_train, X_val, y_train, y_val = map(torch.tensor, train_test_split(X, y, test_size=0.2))
train_dataloader = DataLoader(TensorDataset(X_train.unsqueeze(1), y_train.unsqueeze(1)), batch_size=BATCH_SIZE,
                              pin_memory=True, shuffle=True)
val_dataloader = DataLoader(TensorDataset(X_val.unsqueeze(1), y_val.unsqueeze(1)), batch_size=BATCH_SIZE,
                            pin_memory=True, shuffle=True)

model1 = FunctionApproximator1().to(device)
model2 = FunctionApproximator1().to(device)
optimizer1 = optim.Adam(model1.parameters(), lr=LR_1)
optimizer2 = optim.Adam(model2.parameters(), lr=LR_2)
criterion = nn.MSELoss(reduction="mean")

# training loop for model 1
train_loss_list = []
val_loss_list = []
for epoch in range(MAX_EPOCH):
    print("epoch %d / %d" % (epoch + 1, MAX_EPOCH))
    model1.train()
    # training loop
    temp_loss_list = []
    for X_train, y_train in train_dataloader:
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
    for X_val, y_val in val_dataloader:
        X_val = X_val.type(torch.float32).to(device)
        y_val = y_val.type(torch.float32).to(device)
        score = model1(X_val)
        loss = criterion(input=score, target=y_val)
        temp_loss_list.append(loss.detach().cpu().numpy())
    val_loss_list.append(np.average(temp_loss_list))
    print("  train loss: %.5f" % train_loss_list[-1])
    print("  val loss: %.5f" % val_loss_list[-1])

# training loop for model 2 (trained same way just using different models to hopefully get different result)
train_loss_list = []
val_loss_list = []
for epoch in range(MAX_EPOCH):
    print("epoch %d / %d" % (epoch + 1, MAX_EPOCH))
    model2.train()
    # training loop
    temp_loss_list = []
    for X_train, y_train in train_dataloader:
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
    for X_val, y_val in val_dataloader:
        X_val = X_val.type(torch.float32).to(device)
        y_val = y_val.type(torch.float32).to(device)
        score = model2(X_val)
        loss = criterion(input=score, target=y_val)
        temp_loss_list.append(loss.detach().cpu().numpy())
    val_loss_list.append(np.average(temp_loss_list))

alpha_vals = np.linspace(0,1,10)
fin_tr_loss = []
fin_te_loss = []
fin_tr_acc = []
fin_te_acc = []
for alpha in alpha_vals:
    lin_model = interpolate_models(model1, model2, alpha)
    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    test_acc_list = []
    temp_loss_list = []
    temp_tr_acc_list = []
    temp_te_acc_list = []
    for epoch in range(MAX_EPOCH):
        print("epoch %d / %d" % (epoch + 1, MAX_EPOCH))
        lin_model.train()
        # training loop
        correct = 0
        for X_train, y_train in train_dataloader:
            X_train = X_train.type(torch.float32).to(device)
            y_train = y_train.type(torch.float32).to(device)
            optimizer2.zero_grad()
            score = lin_model(X_train)
            loss = criterion(input=score, target=y_train)
            correct += (score == y_train).sum().item()
            loss.backward()
            optimizer2.step()
            temp_loss_list.append(loss.detach().cpu().numpy())
        train_loss_list.append(np.average(temp_loss_list))
        temp_tr_acc_list.append(correct / len(train_dataloader))
        # validation
        lin_model.eval()
        temp_loss_list = []
        correct = 0
        for X_val, y_val in val_dataloader:
            X_val = X_val.type(torch.float32).to(device)
            y_val = y_val.type(torch.float32).to(device)
            score = lin_model(X_val)
            loss = criterion(input=score, target=y_val)
            correct += (score == y_val).sum().item()
            temp_loss_list.append(loss.detach().cpu().numpy())
        val_loss_list.append(np.average(temp_loss_list))
        temp_te_acc_list.append(correct / len(val_dataloader))
    fin_tr_loss.append(train_loss_list[-1])
    fin_te_loss.append(val_loss_list[-1])
    fin_tr_acc.append(temp_tr_acc_list[-1])
    fin_te_acc.append(temp_te_acc_list[-1])
fig1, (ax1, ax2) = plt.subplots(1,2)

ax1.plot(alpha_vals, fin_tr_loss, color='r', label='train_loss')
ax1.plot(alpha_vals, fin_te_loss, color='g', label='test_loss')
ax1.set_xlabel('alpha')
ax1.set_ylabel('loss')
ax1.legend()

ax2.plot(alpha_vals, fin_tr_acc, color='r', label='train_acc')
ax2.plot(alpha_vals, fin_te_acc, color='g', label='test_acc')
ax2.set_xlabel('alpha')
ax2.set_ylabel('accuracy')
ax2.legend()

fig1.savefig("P3.3.1.png")