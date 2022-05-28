import numpy as np
import torch
import random
from torchvision import transforms

class ToTensorWithoutNormalization:
    def __call__(self, pic):
        pic_np = np.array(pic)
        pic_np = pic_np.transpose((2,0,1))
        pic_tensor = torch.FloatTensor(pic_np)
        return pic_tensor

class ToTensorWithTranspose:
    def __call__(self, pic):
        pic_np = np.array(pic)
        pic_np = pic_np.transpose((2,1,0))
        pic_tensor = torch.FloatTensor(pic_np)
        return pic_tensor

class Random_crop:
    def __call__(self, lr_img, hr_img, hr_crop_size=96, scale=2):
        lr_crop_size = hr_crop_size // scale
        # print(lr_img.shape)
        lr_img_shape = lr_img.shape[-2:]
        # print(lr_img_shape)
        lr_w = int(random.uniform(a=0, b=lr_img_shape[1] - lr_crop_size + 1))
        lr_h = int(random.uniform(a=0, b=lr_img_shape[0] - lr_crop_size + 1))

        hr_w = lr_w * scale
        hr_h = lr_h * scale

        lr_img_cropped = lr_img[:,lr_h:lr_h + lr_crop_size, lr_w:lr_w + lr_crop_size]
        hr_img_cropped = hr_img[:,hr_h:hr_h + hr_crop_size, hr_w:hr_w + hr_crop_size]

        return lr_img_cropped, hr_img_cropped

class Random_flip:
    def __call__(self, lr_img, hr_img):
        H_rate = random.uniform(a=0, b=1)
        V_rate = random.uniform(a=0, b=1)

        if H_rate < 0.5:
            lr_img = transforms.RandomHorizontalFlip(p=1)(lr_img)
            hr_img = transforms.RandomHorizontalFlip(p=1)(hr_img)
        
        if V_rate < 0.5:
            lr_img = transforms.RandomVerticalFlip(p=1)(lr_img)
            hr_img = transforms.RandomVerticalFlip(p=1)(hr_img)
        
        return lr_img, hr_img

class Random_rotate:
    def __call__(self, lr_img, hr_img):
        R_rate = round(random.uniform(a=0, b=4))

        if R_rate == 1:
            lr_img = transforms.RandomRotation(degrees=(90,90), expand=False)(lr_img)
            hr_img = transforms.RandomRotation(degrees=(90,90), expand=False)(hr_img)
        elif R_rate == 2:
            lr_img = transforms.RandomRotation(degrees=(180,180), expand=False)(lr_img)
            hr_img = transforms.RandomRotation(degrees=(180,180), expand=False)(hr_img)
        elif R_rate == 3:
            lr_img = transforms.RandomRotation(degrees=(270,270), expand=False)(lr_img)
            hr_img = transforms.RandomRotation(degrees=(270,270), expand=False)(hr_img)
        
        return lr_img, hr_img