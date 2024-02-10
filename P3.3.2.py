import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


LR_1 = 0.1
LR_2 = 0.01
LR_3 = 0.001
LR_4 = 0.0001
LR_5 = 0.00001
MAX_EPOCH = 5
BATCH_SIZE = 512

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

def sensitivity(y_pred, y_true):
    y_true_classes = np.zeros_like(y_true)
    y_pred_classes = np.zeros_like(y_pred.detach().cpu().numpy())
    y_pred_classes[y_pred > 0.7] = 1
    y_true_classes[y_true > 0.7] = 1
    tn, fp, fn, tp = confusion_matrix(y_true_classes, y_pred_classes).ravel()
    sens = tp / (tp + fn)
    return sens

def accuracy(y_pred, y_true):
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    correct = sum(1 for true, pred in zip(y_true, y_pred) if ((pred >= true - 1) and (pred <= true + 1)))
    total = len(y_true)
    acc = correct / total
    return acc

X = np.random.rand(10 ** 5)
y = simpleFunction(X)

X_train, X_val, y_train, y_val = map(torch.tensor, train_test_split(X, y, test_size=0.2))
train_dataloader = DataLoader(TensorDataset(X_train.unsqueeze(1), y_train.unsqueeze(1)), batch_size=BATCH_SIZE,
                              pin_memory=True, shuffle=True)
val_dataloader = DataLoader(TensorDataset(X_val.unsqueeze(1), y_val.unsqueeze(1)), batch_size=BATCH_SIZE,
                            pin_memory=True, shuffle=True)

models = [FunctionApproximator1().to(device),
    FunctionApproximator1().to(device),
    FunctionApproximator1().to(device),
    FunctionApproximator1().to(device),
    FunctionApproximator1().to(device)]
optimizers = [optim.Adam(models[0].parameters(), lr=LR_1),
    optim.Adam(models[1].parameters(), lr=LR_2),
    optim.Adam(models[2].parameters(), lr=LR_3),
    optim.Adam(models[3].parameters(), lr=LR_4),
    optim.Adam(models[4].parameters(), lr=LR_5)]
criterion = nn.MSELoss(reduction="mean")

fig2, (ax3, ax4) = plt.subplots(1,2)

learning_rates = [0.1,0.01,0.001,0.0001,0.00001]
training_loss = []
test_loss = []
training_acc = []
test_acc = []
sensit = []
for i in range(len(models)):
    # training loop for models
    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    test_acc_list = []
    sens_list = []
    for epoch in range(MAX_EPOCH):
        print("epoch %d / %d" % (epoch + 1, MAX_EPOCH))
        models[i].train()
        # training loop
        temp_loss_list = []
        temp_acc = []
        temp_sens = []
        for X_train, y_train in train_dataloader:
            X_train = X_train.type(torch.float32).to(device)
            y_train = y_train.type(torch.float32).to(device)
            optimizers[i].zero_grad()
            score = models[i](X_train)
            loss = criterion(input=score, target=y_train)
            loss.backward()
            optimizers[i].step()
            temp_sens.append(sensitivity(score, y_train))
            temp_acc.append(accuracy(score, y_train))
            temp_loss_list.append(loss.detach().cpu().numpy())
        train_acc_list.append(np.average(temp_acc))
        train_loss_list.append(np.average(temp_loss_list))
        sens_list.append(np.average(temp_sens))

        # validation
        models[i].eval()
        temp_loss_list = []
        for X_val, y_val in val_dataloader:
            X_val = X_val.type(torch.float32).to(device)
            y_val = y_val.type(torch.float32).to(device)
            score = models[i](X_val)
            loss = criterion(input=score, target=y_val)
            temp_acc.append(accuracy(score, y_val))
            temp_loss_list.append(loss.detach().cpu().numpy())
        val_loss_list.append(np.average(temp_loss_list))
        test_acc_list.append(np.average(temp_acc))
        print("  train loss: %.5f" % train_loss_list[-1])
        print("  val loss: %.5f" % val_loss_list[-1])
    training_loss.append(np.average(train_loss_list))
    test_loss.append(np.average(val_loss_list))
    training_acc.append(np.average(train_acc_list))
    test_acc.append(np.average(test_acc_list))
    sensit.append(np.average(temp_sens))

#accuracy plot for model 1
ax3.plot(learning_rates, training_acc, color='b', label='train_acc')
ax3.plot(learning_rates,test_acc, color='c', label='val_acc')
ax3.plot(learning_rates, sensit, color='r', label='sensitivity')
ax3.legend()
ax3.set_xlabel("learning rate")
ax3.set_ylabel("accuracy")
# loss plot for model 1
ax4.plot(learning_rates, training_loss, color='b', label='train_loss')
ax4.plot(learning_rates,test_loss, color='c', label='val_loss')
ax4.plot(learning_rates, sensit, color='r', label='sensitivity')
ax4.legend()
ax4.set_xlabel("learning rate")
ax4.set_ylabel("loss")
fig2.savefig("P3.3.2.png")
