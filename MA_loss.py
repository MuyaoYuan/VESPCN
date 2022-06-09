import numpy as np
import matplotlib.pyplot as plt

def lossShow_epoch(title, xlabel, ylabel, color, curve_label, savepath, loss_arr, trainset = 800, batch_size = 10, epoch_limit = 200, MA = False):
    divide_span = int(trainset/batch_size/10)
    print(len(loss_arr))
    epochs = min(int(len(loss_arr)/divide_span), epoch_limit)
    epochs_arr = np.arange(epochs) + 1
    loss_arr_epoch = [loss_arr[(epoch) * divide_span - 1] for epoch in epochs_arr]
    if MA:
        pass
    plt.figure()
    plt.plot(epochs_arr, loss_arr_epoch, color, label=curve_label)
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(savepath)

if __name__ == '__main__':
    lossShow_epoch("Learning Curve", "Epochs", "Loss value", "b", "Train Loss", 
                            '1channels.png',
                            'trained_model/ESPCN/ESPCN_train_loss_arr.npy')

