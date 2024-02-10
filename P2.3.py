import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np


LR = 0.0001
MAX_EPOCH = 150
BATCH_SIZE = 512
L2_REGULARIZATION = 0.001
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def simpleFunction(x):
    val = (2 * np.sin(5 * np.pi * x)) / (5 * np.pi * x)
    return val

def gradient_norm_loss(model):
    grad_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm += (p.grad.cpu().data.numpy() ** 2).sum()
    return grad_norm ** 0.5

class SineApproximator(nn.Module):
    def __init__(self):
        super(SineApproximator, self).__init__()
        self.regressor = nn.Sequential(nn.Linear(1, 20),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(20, 20),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(20, 1))

    def forward(self, x):
        output = self.regressor(x)
        return output


X = np.random.rand(10 ** 4)
y = simpleFunction(X)

X_train, X_val, y_train, y_val = map(torch.tensor, train_test_split(X, y, test_size=0.2))
train_dataloader = DataLoader(TensorDataset(X_train.unsqueeze(1), y_train.unsqueeze(1)), batch_size=BATCH_SIZE,
                              pin_memory=True, shuffle=True)
val_dataloader = DataLoader(TensorDataset(X_val.unsqueeze(1), y_val.unsqueeze(1)), batch_size=BATCH_SIZE,
                            pin_memory=True, shuffle=True)

model1 = SineApproximator().to(device)
optimizer1 = optim.Adam(model1.parameters(), lr=LR, weight_decay=L2_REGULARIZATION)
criterion = nn.MSELoss(reduction="mean")
scheduler = optim.lr_scheduler.StepLR(optimizer1, step_size=500, gamma=0.5)
clip_value = 0.5
# training loop with normal loss function
train_loss_list = list()
val_loss_list = list()
for epoch in range(MAX_EPOCH):
    print("epoch %d / %d" % (epoch + 1, MAX_EPOCH))
    model1.train()
    # training loop
    temp_loss_list = list()
    for X_train, y_train in train_dataloader:
        X_train = X_train.type(torch.float32).to(device)
        y_train = y_train.type(torch.float32).to(device)
        optimizer1.zero_grad()
        score = model1(X_train)
        loss = criterion(input=score, target=y_train)
        loss.backward()
        nn.utils.clip_grad_norm_(model1.parameters(), clip_value)
        optimizer1.step()
        temp_loss_list.append(loss.detach().cpu().numpy())
    train_loss_list.append(np.average(temp_loss_list))

    # validation
    model1.eval()
    temp_loss_list = list()
    for X_val, y_val in val_dataloader:
        X_val = X_val.type(torch.float32).to(device)
        y_val = y_val.type(torch.float32).to(device)
        score = model1(X_val)
        loss = criterion(input=score, target=y_val)
        temp_loss_list.append(loss.detach().cpu().numpy())
    val_loss_list.append(np.average(temp_loss_list))
    scheduler.step()
    print("  train loss: %.5f" % train_loss_list[-1])
    print("  val loss: %.5f" % val_loss_list[-1])

min_ratios = []
# training with grad as loss function
for epoch in range(MAX_EPOCH):
    print("epoch %d / %d" % (epoch + 1, MAX_EPOCH))
    model1.train()
    # training loop
    temp_loss_list = list()
    for X_train, y_train in train_dataloader:
        X_train = X_train.type(torch.float32).to(device)
        y_train = y_train.type(torch.float32).to(device)
        optimizer1.zero_grad()
        score = model1(X_train)
        loss = criterion(input=score, target=y_train)
        loss.backward()
        nn.utils.clip_grad_norm_(model1.parameters(), clip_value)
        optimizer1.step()
        temp_loss_list.append(loss.detach().cpu().numpy())
    gradient_norm = gradient_norm_loss(model1)
    print(gradient_norm)
    if gradient_norm < 1e-6:
        Hess = torch.autograd.functional.hessian(loss, model1.parameters())
        eigenvalues, _ = torch.eig(Hess[0])
        minimal_ratio = eigenvalues[:, 0].min() / eigenvalues[:, 0].max()
        min_ratios.append(minimal_ratio.item())
    train_loss_list.append(np.average(temp_loss_list))

    # validation
    model1.eval()
    temp_loss_list = list()
    for X_val, y_val in val_dataloader:
        X_val = X_val.type(torch.float32).to(device)
        y_val = y_val.type(torch.float32).to(device)
        score = model1(X_val)
        loss = criterion(input=score, target=y_val)
        temp_loss_list.append(loss.detach().cpu().numpy())
    val_loss_list.append(np.average(temp_loss_list))
    scheduler.step()
    print("  train loss: %.5f" % train_loss_list[-1])
    print("  val loss: %.5f" % val_loss_list[-1])


model1_y = model1(X_train)
true_y = simpleFunction(X_train)

# plot creation
#fig1, ax1 = plt.subplots(1,2)
# accuracy plot creation for model 1
print(len(min_ratios), " ", len(train_loss_list))
plt.scatter(min_ratios, train_loss_list, color='r')
plt.xlabel("minimum ratio")
plt.ylabel("loss")


plt.savefig("img_p3.png")

