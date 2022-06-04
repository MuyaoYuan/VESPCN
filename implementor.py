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
    def __init__(self, args):
        self.args = args
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
        self.save_path = 'trained_model/' + self.model_name + '/' + self.model_name + '_demo.pkl'
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
    
    def img_SR_YCbCr(self, img_path, save_path):
        img = Image.open(img_path)
        img_width = img.width
        img_height = img.height

        hr_bicubic = img.resize((img_width * self.args.scale, img_height * self.args.scale), resample=Image.BICUBIC)
        hr_bicubic_YCbCr = self.transform(hr_bicubic.convert('YCbCr'))

        img_YCbCr = img.convert('YCbCr')
        img_in_YCbCr = self.transform(img_YCbCr)
        img_in_Y = img_in_YCbCr[0]
        img_in_Y = img_in_Y.to(self.device)
        img_in_Y = img_in_Y.view(1, 1, *img_in_Y.size())

        img_output_Y = self.model(img_in_Y)
        
        img_output_Y = torch.squeeze(img_output_Y)
        img_output_Y = img_output_Y.cpu()
        img_output_YCbCr = torch.stack([img_output_Y, hr_bicubic_YCbCr[1], hr_bicubic_YCbCr[2]])
        img_output_YCbCr = img_output_YCbCr.detach().numpy()
        img_output_YCbCr = np.uint8(img_output_YCbCr*255).transpose((1,2,0))
        img_SR = Image.fromarray(img_output_YCbCr, mode='YCbCr').convert('RGB')

        hr_bicubic.save(os.path.join(save_path, self.model_name + '_BICUBIC.png'))
        img_SR.save(os.path.join(save_path, self.model_name + '_SR.png'))

    def video_SR(self, video_path, save_path):
        cap = cv.VideoCapture(video_path)
        frame_width = int(cap.get(3)) * self.args.scale
        frame_height = int(cap.get(4)) * self.args.scale
        fps = 60
        out = cv.VideoWriter(os.path.join(save_path, self.model_name + '_SR.flv'), cv.VideoWriter_fourcc(*'FLV1'), fps, (frame_width, frame_height))
        while(cap.isOpened()):
            # 获取每一帧图像
            ret, frame = cap.read()
            # 如果获取成功显示图像
            if ret == True:
                # 超分
                '''
                    数组经过行上的切片操作, 会改变数组的连续性。
                    方法一: 复制一份img保存到新的地址
                    img = img[:, :, ::-1]改为img = img[:, :, ::-1].copy()
                    方法二: 将原有的img改为连续的
                    img = img[:, :, ::-1]下一行插入img = np.ascontiguousarray(img)
                    方法三: 直接将原来的numpy.ndarray转为PIL Image格式
                    img = img[:, :, ::-1]下一行插入img = Image.fromarray(np.uint8(img))
                '''
                frame = frame[:,:,::-1].copy() # 将BGR通道调整为RGB通道 加.copy()解决数组连续性问题
                frame_in = self.transform(frame)
                frame_in = frame_in.to(self.device)
                frame_in = frame_in.view(1, *frame_in.size())
                frame_out = self.model(frame_in)
                frame_cv = np.uint8(frame_out.cpu().detach().numpy()*255).transpose(0,2,3,1)
                frame_cv = frame_cv[0][:,:,::-1]
                # 将每一帧写入到输出文件中
                out.write(frame_cv)
            else:
                break
        cap.release()
        out.release()


