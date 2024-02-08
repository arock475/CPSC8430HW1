import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np


LR = 0.0001
MAX_EPOCH = 25
BATCH_SIZE = 512

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def simpleFunction(x):
    val = 2 * np.sin(x)
    #normalized = val / math.sqrt(2)
    return val


class SineApproximator(nn.Module):
    def __init__(self):
        super(SineApproximator, self).__init__()
        self.regressor = nn.Sequential(nn.Linear(1, 1024),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(1024, 1024),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(1024, 1024),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(1024, 1024),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(1024, 1))

    def forward(self, x):
        output = self.regressor(x)
        return output


X = np.random.rand(10 ** 5) * 2 * np.pi
print(X)
y = simpleFunction(X)

X_train, X_val, y_train, y_val = map(torch.tensor, train_test_split(X, y, test_size=0.2))
train_dataloader = DataLoader(TensorDataset(X_train.unsqueeze(1), y_train.unsqueeze(1)), batch_size=BATCH_SIZE,
                              pin_memory=True, shuffle=True)
val_dataloader = DataLoader(TensorDataset(X_val.unsqueeze(1), y_val.unsqueeze(1)), batch_size=BATCH_SIZE,
                            pin_memory=True, shuffle=True)

model1 = SineApproximator().to(device)
optimizer1 = optim.Adam(model1.parameters(), lr=LR)
criterion = nn.MSELoss(reduction="mean")

# training loop for model 1
train_loss_list = list()
val_loss_list = list()
grad_norm_list = []
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
        optimizer1.step()
        temp_loss_list.append(loss.detach().cpu().numpy())
    grad_all = 0
    for p in model1.parameters():
        grad = 0
        if p.grad is not None:
            grad = (p.grad.cpu().data.numpy() ** 2).sum()
        grad_all += grad
    grad_norm_list.append(grad_all ** 0.5)
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
    print("  train loss: %.5f" % train_loss_list[-1])
    print("  val loss: %.5f" % val_loss_list[-1])


model1_y = model1(X_train)
true_y = simpleFunction(X_train)


# plot creation
fig1, (ax1, ax2) = plt.subplots(1,2)
# accuracy plot creation for model 1
ax1.plot(grad_norm_list, color='g', label='val')
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Grad")
ax1.legend()
# loss plot creation for model 1
ax2.plot(train_loss_list, color='r', label='train')
ax2.legend()
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Loss")
ax2.set_title("Training and Val Loss")

fig1.savefig("img_grad.png")