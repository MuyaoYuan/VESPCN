import torch
import torch.nn as nn

class SRCNN(nn.Module):
#upscale_factor -> args
    def __init__(self, n_colors): # n_colors颜色通道数 #scale放大倍数
        super(SRCNN, self).__init__()

        print("Creating SRCNN")

        self.conv1 = nn.Conv2d(n_colors, 64, kernel_size=9, padding=9//2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=1//2)
        self.conv3 = nn.Conv2d(32, n_colors, kernel_size=5, padding=5//2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x