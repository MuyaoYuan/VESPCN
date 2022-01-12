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

def lossShow2(title, xlabel, ylabel, color1, color2, curve_label1, curve_label2, savepath, train_loss_arr, valid_loss_arr):
    n = (np.arange(len(train_loss_arr)) + 1) * 10
    plt.figure()
    plt.plot(n,train_loss_arr, color1, label=curve_label1)
    plt.plot(n,valid_loss_arr, color2, label=curve_label2)
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

def framesProcess(frames_tensor):
    frames_array = frames_tensor.cpu().detach().numpy()
    frames_array = frames_array.transpose((0,1,3,4,2))
    frames_list = []
    for frames_item in frames_array:
        frames = []
        for frame in frames_item:
            frames.append(Image.fromarray(np.uint8(frame*255)))
        frames_list.append(frames)
    return frames_list

def calc_psnr(img1, img2):
    img1 = np.array(img1)
    img2 = np.array(img2)
    mse = np.mean( (img1/255. - img2/255.) ** 2 )
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def psnr_in_valid_dateset(net, datasetLoader, device):
    total_psnr = 0.0
    daset_len = 0
    for i_batch, data_batch in enumerate(datasetLoader):
        # 数据
        inputs = data_batch[0]
        inputs = inputs.to(device)
        labels = data_batch[1]
        labels = labels.to(device)
        outputs = net(inputs)
        # 累加psnr
        # inputs_im = pictureProcess(inputs)
        labels_im = pictureProcess(labels)
        outputs_im = pictureProcess(outputs)
        psnr = calc_psnr(labels_im[0],outputs_im[0])
        total_psnr += psnr
        daset_len += 1
        print(i_batch)
    return total_psnr/daset_len
    

if __name__ == "__main__":
    lossShow("learning curve", "n", "loss value", "b", "train_loss", 
                "result/trainLoss_n.png", np.load("trained_model/train_loss_arr_01.npy"))
    lossShow_epoch("learning curve", "epoch", "loss value", "b", "train_loss", 
                "result/trainLoss_epoch.png", np.load("trained_model/train_loss_arr_01.npy"))
    lossShow2("learning curve", "n", "loss value", "b", "y", "train_loss", "valid_loss",
                "result/Loss_n.png", np.load("trained_model/train_loss_arr_01.npy"), np.load("trained_model/valid_loss_arr_01.npy"))