from PIL import Image
import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.transforms import ToTensor

class vimeo90k(Dataset):
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
        # tensor_label = torch.stack(img_label, dim=0)
        mid_index = int((len(img_label)-1)/2)
        tensor_label = img_label[mid_index]
        return tensor_in, tensor_label, input_path, input_list, label_path, label_list[mid_index]