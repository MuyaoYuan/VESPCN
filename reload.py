import numpy as np
import torch
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from dataset import *
from evaluation import *

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


if __name__ =="__main__":

    # dataset
    """
    # ESPCN & ESPCN_modified
    path_in = "dataset/DIV2K_valid_LR_bicubic_X2"
    path_label = "dataset/DIV2K_valid_HR"
    myDataset = MyDataset(dir_input=path_in,dir_label=path_label,transform=ToTensor())
    """

    # ESPCN_multiframe
    path = "dataset/vimeo90k/vimeo_triplet"
    myDataset = MyDataset_Video(path=path, train=False)

    batch_size=10
    myDataLoader = DataLoader(myDataset, batch_size=batch_size, shuffle=True)

    path_model = "trained_model/ESPCN_multiframe/model_demo.pkl"

    net = torch.load(path_model)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    # print(summary(net,(3, 240, 360)))

    

    print(psnr_in_valid_dateset(net, myDataLoader, device))
    
    # outputs display
    dataIter = iter(myDataLoader)
    dataItem = dataIter.next()
    inputs = dataItem[0]
    inputs = inputs.to(device)
    labels = dataItem[1]
    labels = labels.to(device)
    outputs = net(inputs)
    # ESPCN & ESPCN_modified
    """
    inputs_im = pictureProcess(inputs)
    labels_im = pictureProcess(labels)
    outputs_im = pictureProcess(outputs)
    inputs_im[0].save('result/input.png')
    labels_im[0].save('result/label.png')
    outputs_im[0].save('result/output.png')
    
    print(calc_psnr(labels_im[0],outputs_im[0]))
    """

    # ESPCN_multiframe
    inputs_im = framesProcess(inputs)
    labels_im = pictureProcess(labels)
    outputs_im = pictureProcess(outputs)
    inputs_im[0][0].save('result/demo/input_0.png')
    inputs_im[0][1].save('result/demo/input_1.png')
    inputs_im[0][2].save('result/demo/input_2.png')
    labels_im[0].save('result/demo/label.png')
    outputs_im[0].save('result/demo/output.png')
    print(calc_psnr(labels_im[0],outputs_im[0]))