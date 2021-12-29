from random import sample
from PIL import Image
import os
import sys
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.transforms import ToTensor

class MyDataset(Dataset):
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

"""
在默认情况下，pytorch将图片叠在一起，成为一个N*C*H*W的张量，因此每个batch里的每个图像必须是相同的尺寸。
所以如果想要接受不同尺寸的输入图片，我们就要自己定义collate_fn。
对于图像分类，collate_fn的输入大小是batch_size 大小的list, list里每个元素是一个元组，元组里第一个是图片，第二个是标签。
对于不同大小的输入图片，我们可以使用list来储存。
"""
def my_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    return [data, target]

if __name__ =="__main__":
    # print(os.getcwd()) 查看当前路径
    # print(sys.path[0]) 文件所在路径
    # os.chdir(sys.path[0]) 改变当前路径
    path_in = "dataset/DIV2K_train_LR_bicubic_X2"
    path_label = "dataset/DIV2K_train_HR"
    myDataset = MyDataset(dir_input=path_in,dir_label=path_label,transform=ToTensor())
    myDataLoader = DataLoader(myDataset, batch_size=5, shuffle=True, collate_fn =my_collate)
    for i_batch, items in enumerate(myDataLoader):
        # print(i_batch, items[0].shape, items[1].shape, items[2], items[3])
        print(i_batch, len(items), items[0][1].shape, items[0][2].shape)
        if i_batch == 0:
            break
    # img_in, img_label, title_in, title_label = myDataset[5]
    # print(img_in.shape)
    # print(img_label.shape)
    # print(title_in)
    # print(title_label)