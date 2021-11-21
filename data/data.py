from PIL import Image
import os
import sys
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision.transforms.transforms import ToTensor

class MyDataset(Dataset):
    def __init__(self, dir, transform=None):
        self.dir = dir
        self.file_list = os.listdir(dir)
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.dir, self.file_list[index]))
        if self.transform:
            img = self.transform(img)
        return img, self.file_list[index]

if __name__ =="__main__":
    #print(os.getcwd()) 查看当前路径
    #print(sys.path[0]) 文件所在路径
    #os.chdir(sys.path[0]) 改变当前路径
    path = "dataset/walk"
    myDataset = MyDataset(path,transform=ToTensor())
    img, title = myDataset[1]
    print(img.shape)
    print(title)