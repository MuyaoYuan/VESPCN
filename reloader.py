import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms.transforms import ToTensor
from evaluation import *

# import model
from model.ESPCN import ESPCN
from model.ESPCN_modified import ESPCN_modified
from model.ESPCN_multiframe import ESPCN_multiframe
from model.ESPCN_multiframe2 import ESPCN_multiframe2
from model.motioncompensator import MotionCompensator
from model.VESPCN import VESPCN

# import dataset
from datasetProcess.DIV2K import DIV2K
from datasetProcess.vimeo90k import vimeo90k
from datasetProcess.SRtransforms import ToTensorWithoutNormalization

class Reloader:
    def __init__(self, args, type):
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.type = type
        # select model
        self.model_name = args.model
        if(self.model_name == 'ESPCN'):
            self.model = ESPCN(n_colors=args.n_colors, scale=args.scale).to(self.device)
        elif(self.model_name == 'ESPCN_modified'):
            self.model = ESPCN_modified(n_colors=args.n_colors, scale=args.scale).to(self.device)
        elif(self.model_name == 'ESPCN_multiframe'):
            self.model = ESPCN_multiframe(n_colors=args.n_colors, scale=args.scale, n_sequence=args.n_sequence).to(self.device)
        elif(self.model_name == 'ESPCN_multiframe2'):
            self.model = ESPCN_multiframe2(n_colors=args.n_colors, scale=args.scale, n_sequence=args.n_sequence).to(self.device)
        elif(self.model_name == 'MC'):
            self.model = MotionCompensator(n_colors=args.n_colors, device=self.device).to(self.device)
        elif(self.model_name == 'VESPCN'):
            self.model = VESPCN(n_colors=args.n_colors, scale=args.scale, n_sequence=args.n_sequence, device=self.device).to(self.device)
        else:
            print('Please Enter Appropriate Model!!!')
        # save path
        '''
        保存时代码
            if self.test == True:
                torch.save(self.model, 'trained_model/' + self.model_name + '/'+self.model_name + '_demo.pkl')
                np.save('trained_model/' + self.model_name + '/' + self.model_name +"_train_loss_arr_demo.npy",train_loss_arr)
                np.save('trained_model/' + self.model_name + '/' + self.model_name + "_valid_loss_arr_demo.npy",valid_loss_arr)
            else:
                torch.save(self.model, 'trained_model/' + self.model_name + '/' + self.model_name + '.pkl')
                np.save('trained_model/' + self.model_name + '/' + self.model_name +"_train_loss_arr.npy",train_loss_arr)
                np.save('trained_model/' + self.model_name + '/' + self.model_name + "_valid_loss_arr.npy",valid_loss_arr)
        '''
        if type == 'pre':
            self.save_path = 'trained_model/' + self.model_name + '/'+ self.model_name + '_demo.pkl'
        elif type == 'trained':
            self.save_path = 'trained_model/' + self.model_name + '/' + self.model_name + '.pkl'
        else:
            print('Please Enter Appropriate Reload Type!!!')

        # model reload
        # print(self.save_path)
        self.model.load_state_dict(torch.load(self.save_path))
        # transform select
        if(args.transform == 'null'):
            self.transform = ToTensor()
            self.Normalization = True
        elif(args.transform == 'withoutNormalization'):
            self.transform = ToTensorWithoutNormalization()
            self.Normalization = False

        else:
            print('Please Enter Appropriate Transform!!!')
        # Dateset select
        self.dataset_name = args.dataset_name
        if(args.dataset_name == 'DIV2K'):
            self.train_path_in = args.train_path_in
            self.train_path_label = args.train_path_label
            self.valid_path_in = args.valid_path_in
            self.valid_path_label = args.valid_path_label
            self.batch_size = args.batch_size 
            self.trainDataset = DIV2K(dir_input=self.train_path_in,dir_label=self.train_path_label,transform=self.transform)
            self.trainDataLoader = DataLoader(self.trainDataset, batch_size=self.batch_size, shuffle=True)
            self.validDataset = DIV2K(dir_input=self.valid_path_in,dir_label=self.valid_path_label,transform=self.transform)
            self.validDataLoader = DataLoader(self.validDataset, batch_size=self.batch_size, shuffle=True)
        elif(args.dataset_name == 'vimeo90k'):
            self.dataset_path = args.dataset_path
            self.batch_size = args.batch_size
            self.trainDataset = vimeo90k(path = self.dataset_path, transform = self.transform)
            self.trainDataLoader = DataLoader(self.trainDataset, batch_size=self.batch_size, shuffle=True)
            self.validDataset = vimeo90k(path = self.dataset_path, train=False, transform = self.transform)
            self.validDataLoader = DataLoader(self.validDataset, batch_size=self.batch_size, shuffle=True)
        else:
            print('Please Enter Appropriate Dataset!!!')
    
    def print_psnr_in_valid_dateset(self):
        print('psnr in valid dateset:' + str(psnr_in_valid_dateset(self.model, self.validDataLoader, self.device)))
    
    def outputs_display(self):
        dataIter = iter(self.validDataLoader)
        dataItem = dataIter.next()
        inputs = dataItem[0]
        inputs = inputs.to(self.device)
        labels = dataItem[1]
        labels = labels.to(self.device)
        outputs = self.model(inputs)
        if(self.dataset_name == 'DIV2K'):
            inputs_im = pictureProcess(inputs,self.Normalization)
            labels_im = pictureProcess(labels,self.Normalization)
            outputs_im = pictureProcess(outputs,self.Normalization)
            if self.type == 'pre':
                inputs_im[0].save('result/' + self.model_name + '/demo/input_demo.png')
                labels_im[0].save('result/' + self.model_name + '/demo/label_demo.png')
                outputs_im[0].save('result/' + self.model_name + '/demo/output_demo.png')
            elif self.type == 'trained':
                inputs_im[0].save('result/' + self.model_name + '/demo/input.png')
                labels_im[0].save('result/' + self.model_name + '/demo/label.png')
                outputs_im[0].save('result/' + self.model_name + '/demo/output.png')
        if(self.dataset_name == 'vimeo90k'):
            inputs_im = framesProcess(inputs,self.Normalization)
            labels_im = pictureProcess(labels,self.Normalization)
            outputs_im = pictureProcess(outputs,self.Normalization)
            if self.type == 'pre':
                inputs_im[0][0].save('result/' + self.model_name + '/demo/input_0_demo.png')
                inputs_im[0][1].save('result/' + self.model_name + '/demo/input_1_demo.png')
                inputs_im[0][2].save('result/' + self.model_name + '/demo/input_2_demo.png')
                labels_im[0].save('result/' + self.model_name + '/demo/label_demo.png')
                outputs_im[0].save('result/' + self.model_name + '/demo/output_demo.png')
            elif(self.type == 'trained'):
                inputs_im[0][0].save('result/' + self.model_name + '/demo/input_0.png')
                inputs_im[0][1].save('result/' + self.model_name + '/demo/input_1.png')
                inputs_im[0][2].save('result/' + self.model_name + '/demo/input_2.png')
                labels_im[0].save('result/' + self.model_name + '/demo/label.png')
                outputs_im[0].save('result/' + self.model_name + '/demo/output.png')
        print('psnr of demo: ' + str(calc_psnr(labels_im[0],outputs_im[0])))
    
    def loss_display(self):
        if(self.type == 'pre'):
            lossShow("learning curve", "batches", "loss value", "b", "train_loss", 
                    'result/' + self.model_name + '/loss/trainLoss_batches_demo.png', 
                    np.load('trained_model/' + self.model_name + '/' + self.model_name +"_train_loss_arr_demo.npy"))
            # lossShow_epoch("learning curve", "epoch", "loss value", "b", "train_loss", 
            #             "result/loss/trainLoss_epoch.png", np.load("trained_model/ESPCN_multiframe/train_loss_arr.npy"))
            lossShow2("learning curve", "batches", "loss value", "b", "y", "train_loss", "valid_loss",
                        'result/' + self.model_name + '/loss/Loss_batches_demo.png', 
                        np.load('trained_model/' + self.model_name + '/' + self.model_name +"_train_loss_arr_demo.npy"), 
                        np.load('trained_model/' + self.model_name + '/' + self.model_name +"_valid_loss_arr_demo.npy"))
        elif(self.type == 'trained'):
            # print('lossShow')
            lossShow("learning curve", "batches", "loss value", "b", "train_loss", 
                    'result/' + self.model_name + '/loss/trainLoss_batches.png', 
                    np.load('trained_model/' + self.model_name + '/' + self.model_name +"_train_loss_arr.npy"))
            # lossShow_epoch("learning curve", "epoch", "loss value", "b", "train_loss", 
            #             "result/loss/trainLoss_epoch.png", np.load("trained_model/ESPCN_multiframe/train_loss_arr.npy"))
            lossShow2("learning curve", "batches", "loss value", "b", "y", "train_loss", "valid_loss",
                        'result/' + self.model_name + '/loss/Loss_batches.png', 
                        np.load('trained_model/' + self.model_name + '/' + self.model_name +"_train_loss_arr.npy"), 
                        np.load('trained_model/' + self.model_name + '/' + self.model_name +"_valid_loss_arr.npy"))

    def outputs_display_MC(self):
        dataIter = iter(self.validDataLoader)
        dataItem = dataIter.next()
        frames = dataItem[0]
        frame1 = frames[:,1]
        frame1 = frame1.to(self.device)
        frame2 = frames[:,0]
        frame2 = frame2.to(self.device)

        frame2_compensated, flow = self.model(frame1, frame2)

        if(self.dataset_name == 'vimeo90k'):
            frame1_im = pictureProcess(frame1)
            frame2_im = pictureProcess(frame2)
            frame2_compensated_im = pictureProcess(frame2_compensated)
            if self.type == 'pre':
                frame1_im[0].save('result/' + self.model_name + '/demo/frame1_demo.png')
                frame2_im[0].save('result/' + self.model_name + '/demo/frame2_demo.png')
                frame2_compensated_im[0].save('result/' + self.model_name + '/demo/frame2_compensated_demo.png')
            elif(self.type == 'trained'):
                frame1_im[0].save('result/' + self.model_name + '/demo/frame1.png')
                frame2_im[0].save('result/' + self.model_name + '/demo/frame2.png')
                frame2_compensated_im[0].save('result/' + self.model_name + '/demo/frame2_compensated.png')
        print('psnr of demo: ' + str(calc_psnr(frame1_im[0],frame2_compensated_im[0])))
    
    def outputs_display_VESPCN(self):
        dataIter = iter(self.validDataLoader)
        dataItem = dataIter.next()
        inputs = dataItem[0]
        inputs = inputs.to(self.device)
        labels = dataItem[1]
        labels = labels.to(self.device)
        outputs, _, _ = self.model(inputs)
        if(self.dataset_name == 'vimeo90k'):
            inputs_im = framesProcess(inputs)
            labels_im = pictureProcess(labels)
            outputs_im = pictureProcess(outputs)
            if self.type == 'pre':
                inputs_im[0][0].save('result/' + self.model_name + '/demo/input_0_demo.png')
                inputs_im[0][1].save('result/' + self.model_name + '/demo/input_1_demo.png')
                inputs_im[0][2].save('result/' + self.model_name + '/demo/input_2_demo.png')
                labels_im[0].save('result/' + self.model_name + '/demo/label_demo.png')
                outputs_im[0].save('result/' + self.model_name + '/demo/output_demo.png')
            elif(self.type == 'trained'):
                inputs_im[0][0].save('result/' + self.model_name + '/demo/input_0.png')
                inputs_im[0][1].save('result/' + self.model_name + '/demo/input_1.png')
                inputs_im[0][2].save('result/' + self.model_name + '/demo/input_2.png')
                labels_im[0].save('result/' + self.model_name + '/demo/label.png')
                outputs_im[0].save('result/' + self.model_name + '/demo/output.png')
        print('psnr of demo: ' + str(calc_psnr(labels_im[0],outputs_im[0])))
    