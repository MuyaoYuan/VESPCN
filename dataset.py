from random import sample
from PIL import Image
import os
import sys
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.transforms import ToTensor

class MyDataset(Dataset):
    def __init__(self, dir_input, dir_label, transform=None):
        self.dir_input = dir_input
        self.dir_label = dir_label
        self.file_list_input = os.listdir(dir_input)
        self.file_list_label = os.listdir(dir_label)
        self.transform = transform

    def __len__(self):
        return len(self.file_list_label)

    def __getitem__(self, index):
        img_in = Image.open(os.path.join(self.dir_input, self.file_list_input[index]))
        img_label = Image.open(os.path.join(self.dir_label, self.file_list_label[index]))
        if self.transform:
            img_in = self.transform(img_in)
            img_label = self.transform(img_label)
        return img_in, img_label, self.file_list_label[index]

if __name__ =="__main__":
    # print(os.getcwd()) 查看当前路径
    # print(sys.path[0]) 文件所在路径
    # os.chdir(sys.path[0]) 改变当前路径
    path_in = "dataset/walk_downscale"
    path_label = "dataset/walk"
    myDataset = MyDataset(dir_input=path_in,dir_label=path_label,transform=ToTensor())
    myDataLoader = DataLoader(myDataset, batch_size=5, shuffle=True)
    for i_batch, items in enumerate(myDataLoader):
        print(i_batch, items[0].shape, items[1].shape, items[2])
        if i_batch == 2:
            break
    # img_in, img_label, title = myDataset[5]
    # print(img_in.shape)
    # print(img_label.shape)
    # print(title)