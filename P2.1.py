import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np


LR = 0.0001
MAX_EPOCH = 4
BATCH_SIZE = 512

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def simpleFunction(x):
    val = (np.sin(5 * np.pi * x)) / (5 * np.pi * x)
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

X = np.random.rand(10 ** 5)
y = simpleFunction(X)


X_train, X_val, y_train, y_val = map(torch.tensor, train_test_split(X, y, test_size=0.2))
train_dataloader = DataLoader(TensorDataset(X_train.unsqueeze(1), y_train.unsqueeze(1)), batch_size=BATCH_SIZE,
                              pin_memory=True, shuffle=True)
val_dataloader = DataLoader(TensorDataset(X_val.unsqueeze(1), y_val.unsqueeze(1)), batch_size=BATCH_SIZE,
                            pin_memory=True, shuffle=True)

model1 = FunctionApproximator1().to(device)
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
        weights = []
        if epoch % 3 == 0:
            for param_tensor in model1.state_dict():
                weights.append(model1.state_dict()[param_tensor].numpy().flatten())
            vis_weight_temp.append(np.concatenate(weights))

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
    vis_weights.append(vis_weight_temp)
    epochs.append(epochs_temp)


model1_y = model1(X_train)
true_y = simpleFunction(X_train)



# plot creation for loss
fig1, (ax1, ax2) = plt.subplots(1,2)
# accuracy plot creation for layer 0
ax1.scatter(epochs[0], vis_loss[0], color='r',label='layer 0')
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.legend()
# loss plot creation for whole model
for i in range(8):
    ax2.scatter(epochs[0], vis_loss[i], color='r')
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")

fig1.savefig("img2.1_vis.png")

# plot creation for weights
fig2, (ax3, ax4) = plt.subplots(1,2)
#print(vis_weights)

PCA_weights = []


weights_array = np.array(vis_weights)

pca = PCA(n_components=2)

epochs = weights_array.shape[0]
weights_reshaped = weights_array.reshape(MAX_EPOCH, -1)

# Fit PCA to the reshaped weights
pca.fit(weights_array)

# Transform the weights to 2D using PCA
weights_pca = pca.transform(weights_reshaped)

# Now, weights_pca contains the weights collected every 3 epochs from training, reduced to 2 dimensions using PCA





# accuracy plot creation for layer 0
ax3.scatter(epochs[0], vis_loss[0], color='r',label='Model 1')
ax3.set_xlabel("PC1")
ax3.set_ylabel("PC2")
ax3.legend()
# loss plot creation for whole model
for i in range(8):
    ax2.scatter(epochs[0], vis_loss[i], color='r')
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")

fig1.savefig("img2.1_weight.png")