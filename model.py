import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torchsummary import summary

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

class ESPCN(nn.Module):
#upscale_factor -> args
    def __init__(self, n_colors, scale): # n_colors颜色通道数 #scale放大倍数
        super(ESPCN, self).__init__()

        print("Creating ESPCN (x%d)" %scale)

        self.conv1 = nn.Conv2d(n_colors, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, n_colors * scale * scale, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale) # 将低分辨率图像变成高分辨率
        self.conv4 = nn.Conv2d(n_colors, n_colors, kernel_size=1, padding=0)
        self.relu = nn.ReLU()
        # self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(x)
        x = self.conv4(x)
        x = self.sigmoid(x)
        return x

if __name__ =="__main__":
    net = ESPCN(n_colors=3, scale=2)
    print(summary(net, (3, 240, 360)))