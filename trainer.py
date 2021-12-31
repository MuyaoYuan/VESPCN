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
    # net = ESPCN(n_colors=3, scale=2)
    net = ESPCN_modified(n_colors=3, scale=2)
    net.to(device)

    # 数据集
    train_path_in = "dataset/DIV2K_train_LR_bicubic_X2"
    train_path_label = "dataset/DIV2K_train_HR"
    valid_path_in = "dataset/DIV2K_valid_LR_bicubic_X2"
    valid_path_label = "dataset/DIV2K_valid_HR"
    batch_size = 1
    trainDataset = MyDataset(dir_input=train_path_in,dir_label=train_path_label,transform=ToTensor())
    # myDataLoader = DataLoader(myDataset, batch_size=batch_size, shuffle=True, collate_fn =my_collate)
    trainDataLoader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True)
    validDataset = MyDataset(dir_input=valid_path_in,dir_label=valid_path_label,transform=ToTensor())
    validDataLoader = DataLoader(validDataset, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()
    # criterion = F.binary_cross_entropy
    optimizer = optim.Adam(net.parameters(), lr=1e-4)

    init_weights(net)

    running_loss = 0.0
    valid_loss = 0.0
    train_loss_arr = np.array([])
    valid_loss_arr = np.array([])
    epochs = 50
    for epoch in range(epochs):
        print(epoch)
        for i_batch, data_batch in enumerate(trainDataLoader):
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
                # print('[%d, %5d] trainLoss:%.3f' %
                #         (epoch + 1, i_batch + 1, running_loss/10))
                # train_loss
                train_loss_arr = np.append(train_loss_arr,running_loss/10)
                running_loss = 0.0
                # valid_loss
                validDataIter = iter(validDataLoader)
                for i in range(10):
                    data_valid = validDataIter.next()
                    inputs_valid = data_batch[0]
                    inputs_valid = inputs_valid.to(device)
                    labels_valid = data_batch[1]
                    labels_valid = labels_valid.to(device)
                    outputs_valid = net(inputs)
                    loss_valid = criterion(outputs_valid, labels_valid)
                    valid_loss += loss_valid.item()
                # print('[%d, %5d] validLoss:%.3f' %
                #         (epoch + 1, i_batch + 1, valid_loss/10))
                valid_loss_arr = np.append(valid_loss_arr,valid_loss/10)
                valid_loss = 0.0

    print("Finished Training")
    torch.save(net, 'trained_model/model_demo_01.pkl')
    np.save("trained_model/train_loss_arr_01.npy",train_loss_arr)
    np.save("trained_model/valid_loss_arr_01.npy",valid_loss_arr)

            


