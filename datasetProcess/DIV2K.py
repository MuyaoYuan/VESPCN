from PIL import Image
import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.transforms import ToTensor

class DIV2K(Dataset):
    def __init__(self, dir_input, dir_label, transform=ToTensor()):
        self.dir_input = dir_input
        self.dir_label = dir_label

        file_list_input = os.listdir(dir_input)
        file_list_input.sort(key=lambda x:int(x[:-6]))
        self.file_list_input = file_list_input

        file_list_label = os.listdir(dir_label)
        file_list_label.sort(key=lambda x:int(x[:-4])) #倒着数第四位'.'为分界线，按照‘.'左边的数字从小到大排序
        self.file_list_label = file_list_label

        self.transform = transform

    def __len__(self):
        return len(self.file_list_label)

    def __getitem__(self, index):
        img_in = Image.open(os.path.join(self.dir_input, self.file_list_input[index]))
        img_label = Image.open(os.path.join(self.dir_label, self.file_list_label[index]))
        if self.transform:
            img_in = self.transform(img_in)
            img_label = self.transform(img_label)
        return img_in, img_label, self.file_list_input[index], self.file_list_label[index]