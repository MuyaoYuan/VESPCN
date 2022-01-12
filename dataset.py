from random import sample
from PIL import Image
import os
import sys
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.transforms import ToTensor

from evaluation import *

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

class MyDataset_Video(Dataset):
    def __init__(self, path, train=True, transform=ToTensor()):
        self.path = path
        if train:
            self.txtfile = os.path.join(self.path, "tri_trainlist.txt")
        else:
            self.txtfile = os.path.join(self.path, "tri_testlist.txt")

        with open(self.txtfile) as f:
            datasetList = f.readlines()
            self.datasetList = [item.strip() for item in datasetList]
            
        self.transform = transform


    def __len__(self):
        return len(self.datasetList)

    def __getitem__(self, index):
        input_path = os.path.join(self.path, "sequences_LR_bicubic_X2", self.datasetList[index])
        label_path = os.path.join(self.path, "sequences", self.datasetList[index])

        input_list = os.listdir(input_path)
        input_list.sort(key=lambda x:int(x[-5:-4])) #倒着数第四位'.'为分界线，按照‘.'左边的数字从小到大排序
        label_list = os.listdir(label_path)
        label_list.sort(key=lambda x:int(x[-5:-4])) #倒着数第四位'.'为分界线，按照‘.'左边的数字从小到大排序

        img_in = []
        img_label = []

        for i in range(len(input_list)):
            img_in_item = Image.open(os.path.join(input_path, input_list[i]))
            img_label_item = Image.open(os.path.join(label_path, label_list[i]))
            if self.transform:
                img_in_item = self.transform(img_in_item)
                img_label_item = self.transform(img_label_item)
            img_in.append(img_in_item)
            img_label.append(img_label_item)

        tensor_in = torch.stack(img_in, dim=0)
        tensor_label = torch.stack(img_label, dim=0)
        return tensor_in, tensor_label, input_path, input_list, label_path, label_list

        

"""
在默认情况下，pytorch将图片叠在一起，成为一个N*C*H*W的张量，因此每个batch里的每个图像必须是相同的尺寸。
所以如果想要接受不同尺寸的输入图片，我们就要自己定义collate_fn。
对于图像分类，collate_fn的输入大小是batch_size 大小的list, list里每个元素是一个元组，元组里第一个是图片，第二个是标签。
对于不同大小的输入图片，我们可以使用list来储存。
"""
# def my_collate(batch):
#     data = [item[0] for item in batch]
#     target = [item[1] for item in batch]
#     return [data, target]

if __name__ =="__main__":
    # print(os.getcwd()) 查看当前路径
    # print(sys.path[0]) 文件所在路径
    # os.chdir(sys.path[0]) 改变当前路径

    """
    path_in = "dataset/DIV2K_train_LR_bicubic_X2"
    path_label = "dataset/DIV2K_train_HR"
    myDataset = MyDataset(dir_input=path_in,dir_label=path_label,transform=ToTensor())
    # myDataLoader = DataLoader(myDataset, batch_size=5, shuffle=True, collate_fn =my_collate)
    myDataLoader = DataLoader(myDataset, batch_size=1, shuffle=True)
    for i_batch, items in enumerate(myDataLoader):
        print(i_batch, items[0].shape, items[1].shape, items[2], items[3])
        # print(i_batch, len(items), items[0][1].shape, items[0][2].shape)
        if i_batch == 0:
            break
    # img_in, img_label, title_in, title_label = myDataset[5]
    # print(img_in.shape)
    # print(img_label.shape)
    # print(title_in)
    # print(title_label)
    """

    path = "dataset/vimeo90k/vimeo_triplet"
    batch_size = 1
    myDataset = MyDataset_Video(path = path)
    myDataLoader = DataLoader(myDataset, batch_size=batch_size, shuffle=True)

    dataIter = iter(myDataLoader)
    dataItem = dataIter.next()
    
    print(dataItem[0].shape, dataItem[1].shape, dataItem[2], dataItem[3], dataItem[4], dataItem[5])

    inputs = framesProcess(dataItem[0])
    labels = framesProcess(dataItem[1])
    inputs[0][0].save("input_0.png")
    inputs[0][1].save("input_1.png")
    inputs[0][2].save("input_2.png")
    labels[0][0].save("label_0.png")
    labels[0][1].save("label_1.png")
    labels[0][2].save("label_2.png")


