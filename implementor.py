from PIL import Image
import os
import torch
from torchvision.transforms.transforms import ToTensor
import cv2 as cv

from model.ESPCN import ESPCN
from model.ESPCN_modified import ESPCN_modified
from model.ESPCN_multiframe import ESPCN_multiframe
from evaluation import *


class Implementor:
    def __init__(self, args, type):
        self.args = args
        self.type = type
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # select model
        self.model_name = args.model
        if(self.model_name == 'ESPCN'):
            self.model = ESPCN(n_colors=args.n_colors, scale=args.scale).to(self.device)
        elif(self.model_name == 'ESPCN_modified'):
            self.model = ESPCN_modified(n_colors=args.n_colors, scale=args.scale).to(self.device)
        elif(self.model_name == 'ESPCN_multiframe'):
            self.model = ESPCN_multiframe(n_colors=args.n_colors, scale=args.scale, n_sequence=args.n_sequence).to(self.device)
        else:
            print('Please Enter Appropriate Model!!!')
        # save path
        self.save_path = 'trained_model/' + self.model_name + '/' + self.model_name + '.pkl'
        # model reload
        # print(self.save_path)
        self.model.load_state_dict(torch.load(self.save_path))
        self.transform = ToTensor()

    def img_SR(self, img_path, save_path):
        img = Image.open(img_path)
        img_in = self.transform(img)
        img_in = img_in.to(self.device)
        img_in = img_in.view(1, *img_in.size())
        img_output = self.model(img_in)
        img_SR = pictureProcess(img_output)
        img_SR[0].save(os.path.join(save_path, self.model_name + '_SR.png'))


