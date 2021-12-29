import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
from dataset import *

def pictureProcess(picture_tensor):
    picture_array = picture_tensor.detach().numpy()
    picture_array = picture_array.transpose((0,2,3,1))
    im_list = []
    for picture_item in picture_array:
        im_list.append(Image.fromarray(np.uint8(picture_item*255)))
    return im_list

if __name__ =="__main__":
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
    
    inputs_im = pictureProcess(inputs)
    labels_im = pictureProcess(labels)
    outputs_im = pictureProcess(outputs)
    inputs_im[1].save('result/input.png')
    labels_im[1].save('result/label.png')
    outputs_im[1].save('result/output.png')