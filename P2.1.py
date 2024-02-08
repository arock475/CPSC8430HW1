import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np


LR = 0.0001
MAX_EPOCH = 4
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
vis_weights = []
vis_loss = []
epochs = []
for i in range(8):
    vis_loss_temp = []
    vis_weight_temp = []
    epochs_temp = []
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
        if epoch % 3 == 0:
            #vis_weight_temp.append()
            epochs_temp.append(epoch)
            vis_loss_temp.append(np.average(temp_loss_list))
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
    vis_loss.append(vis_loss_temp)
    #vis_weights.append(vis_weight_temp)
    epochs.append(epochs_temp)


model1_y = model1(X_train)
true_y = simpleFunction(X_train)

#print(vis_weight_temp)
print(vis_loss)

# plot creation for loss
fig1, (ax1, ax2) = plt.subplots(1,2)
# accuracy plot creation for layer 0
ax1.scatter(epochs[0], vis_loss[0], color='r',label='layer 0')
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.legend()
# loss plot creation for whole model
for i in range(8):
    ax2.scatter(epochs[0], vis_loss[i], color='r', label='whole model')
ax2.legend()
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")

fig1.savefig("img_vis.png")