import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader

from model import *
from dataset import *

# 网络模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = ESPCN(n_colors=3, scale=2)
net.to(device)

# 数据集
path_in = "dataset/walk_downscale"
path_label = "dataset/walk"
myDataset = MyDataset(dir_input=path_in,dir_label=path_label,transform=ToTensor())
myDataLoader = DataLoader(myDataset, batch_size=5, shuffle=True)

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

init_weights(net)

running_loss = 0.0
for epoch in range(5):
    for i_batch, data_batch in enumerate(myDataLoader):
        # 数据
        inputs = data_batch[0]
        labels = data_batch[1]
        # 一次梯度下降
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # train_loss
        running_loss = loss.item()
        print('[%d, %5d] loss:%.3f' %
                  (epoch + 1, i_batch + 1, running_loss))
print("Finished Training")
torch.save(net, 'trained_model/model_demo.pkl')
        

        


