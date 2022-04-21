import numpy as np
import torch

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