import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math

def lossShow(title, xlabel, ylabel, color, curve_label, savepath, loss_arr):
    n = (np.arange(len(loss_arr)) + 1) * 10
    plt.figure()
    plt.plot(n,loss_arr, color, label=curve_label)
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(savepath)

def lossShow_epoch(title, xlabel, ylabel, color, curve_label, savepath, loss_arr, trainset = 800):
    trainset = int(trainset/10)
    epochs = int(len(loss_arr)/trainset)
    epochs_arr = np.arange(epochs) + 1
    loss_arr_epoch = [loss_arr[(epoch) * trainset - 1] for epoch in epochs_arr]
    plt.figure()
    plt.plot(epochs_arr, loss_arr_epoch, color, label=curve_label)
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(savepath)

def pictureProcess(picture_tensor):
    picture_array = picture_tensor.cpu().detach().numpy()
    picture_array = picture_array.transpose((0,2,3,1))
    im_list = []
    for picture_item in picture_array:
        im_list.append(Image.fromarray(np.uint8(picture_item*255)))
    return im_list

def calc_psnr(img1, img2):
    img1 = np.array(img1)
    img2 = np.array(img2)
    mse = np.mean( (img1/255. - img2/255.) ** 2 )
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

if __name__ == "__main__":
    lossShow("learning curve", "n", "loss value", "b", "train_loss", 
                "result/trainLoss_n.png", np.load("trained_model/train_loss_arr.npy"))
    lossShow_epoch("learning curve", "epoch", "loss value", "b", "train_loss", 
                "result/trainLoss_epoch.png", np.load("trained_model/train_loss_arr.npy"))