import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from model import *
from dataset import *

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ =="__main__":
    # 网络模型
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = ESPCN(n_colors=3, scale=2)
    net.to(device)

    # 数据集
    path_in = "dataset/DIV2K_train_LR_bicubic_X2"
    path_label = "dataset/DIV2K_train_HR"
    batch_size = 1
    myDataset = MyDataset(dir_input=path_in,dir_label=path_label,transform=ToTensor())
    # myDataLoader = DataLoader(myDataset, batch_size=batch_size, shuffle=True, collate_fn =my_collate)
    myDataLoader = DataLoader(myDataset, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()
    # criterion = F.binary_cross_entropy
    optimizer = optim.Adam(net.parameters(), lr=1e-4)

    init_weights(net)

    running_loss = 0.0
    train_loss_arr = np.array([])
    epochs = 50
    for epoch in range(epochs):
        for i_batch, data_batch in enumerate(myDataLoader):
            # 数据
            inputs = data_batch[0]
            inputs = inputs.to(device)
            labels = data_batch[1]
            labels = labels.to(device)
            # 一次梯度下降
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # train_loss
            running_loss += loss.item()
            if i_batch % 10 == 9:
                # print('[%d, %5d] loss:%.3f' %
                #         (epoch + 1, i_batch + 1, running_loss/10))
                train_loss_arr = np.append(train_loss_arr,running_loss/10)
                running_loss = 0.0
    print("Finished Training")
    torch.save(net, 'trained_model/model_demo.pkl')
    np.save("trained_model/train_loss_arr.npy",train_loss_arr)

            


