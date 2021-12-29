import numpy as np
import torch
import matplotlib.pyplot as plt
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
from dataset import *

path_in = "dataset/walk_downscale"
path_label = "dataset/walk"
path_model = "trained_model/model_demo.pkl"

net = torch.load(path_model)
# print(summary(net,(3, 240, 360)))

myDataset = MyDataset(dir_input=path_in,dir_label=path_label,transform=ToTensor())
myDataLoader = DataLoader(myDataset, batch_size=5, shuffle=True)

for i_batch, data_batch in enumerate(myDataLoader):
    inputs = data_batch[0]
    labels = data_batch[1]
    outputs = net(inputs)
    if i_batch == 0:
        break

plt.subplot(1,3,1)
plt.imshow(inputs[1].detach().numpy().T)
# print(np.max(inputs[1].detach().numpy()))
# print(np.min(inputs[1].detach().numpy()))

plt.subplot(1,3,2)
plt.imshow(labels[1].detach().numpy().T)
# print(np.max(labels[1].detach().numpy()))
# print(np.min(labels[1].detach().numpy()))

plt.subplot(1,3,3)
plt.imshow(outputs[1].detach().numpy().T)
# print(outputs[1].detach().numpy().T.shape)
# print(np.max(outputs[1].detach().numpy()))
# print(np.min(outputs[1].detach().numpy()))
plt.show()