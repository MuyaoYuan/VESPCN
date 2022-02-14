import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms.transforms import ToTensor
from evaluation import *

# import dataset
from datasetProcess.DIV2K import DIV2K
from datasetProcess.vimeo90k import vimeo90k

class Reloader:
    def __init__(self, args, type):
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_name = args.model
        self.type = type
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
        self.model = torch.load(self.save_path).to(self.device)
        # Dateset select
        self.dataset_name = args.dataset_name
        if(args.dataset_name == 'DIV2K'):
            self.train_path_in = args.train_path_in
            self.train_path_label = args.train_path_label
            self.valid_path_in = args.valid_path_in
            self.valid_path_label = args.valid_path_label
            self.batch_size = args.batch_size 
            self.trainDataset = DIV2K(dir_input=self.train_path_in,dir_label=self.train_path_label,transform=ToTensor())
            self.trainDataLoader = DataLoader(self.trainDataset, batch_size=self.batch_size, shuffle=True)
            self.validDataset = DIV2K(dir_input=self.valid_path_in,dir_label=self.valid_path_label,transform=ToTensor())
            self.validDataLoader = DataLoader(self.validDataset, batch_size=self.batch_size, shuffle=True)
        elif(args.dataset_name == 'vimeo90k'):
            self.dataset_path = args.dataset_path
            self.batch_size = args.batch_size
            self.trainDataset = vimeo90k(path = self.dataset_path)
            self.trainDataLoader = DataLoader(self.trainDataset, batch_size=self.batch_size, shuffle=True)
            self.validDataset = vimeo90k(path = self.dataset_path, train=False)
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
            inputs_im = pictureProcess(inputs)
            labels_im = pictureProcess(labels)
            outputs_im = pictureProcess(outputs)
            inputs_im[0].save('result/' + self.model_name + '/demo/input.png')
            labels_im[0].save('result/' + self.model_name + '/demo/label.png')
            outputs_im[0].save('result/' + self.model_name + '/demo/output.png')
        if(self.dataset_name == 'vimeo90k'):
            inputs_im = framesProcess(inputs)
            labels_im = pictureProcess(labels)
            outputs_im = pictureProcess(outputs)
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
            lossShow("learning curve", "batches", "loss value", "b", "train_loss", 
                    'result/' + self.model_name + '/loss/trainLoss_batches.png', 
                    np.load('trained_model/' + self.model_name + '/' + self.model_name +"_train_loss_arr.npy"))
            # lossShow_epoch("learning curve", "epoch", "loss value", "b", "train_loss", 
            #             "result/loss/trainLoss_epoch.png", np.load("trained_model/ESPCN_multiframe/train_loss_arr.npy"))
            lossShow2("learning curve", "batches", "loss value", "b", "y", "train_loss", "valid_loss",
                        'result/' + self.model_name + '/loss/Loss_batches.png', 
                        np.load('trained_model/' + self.model_name + '/' + self.model_name +"_train_loss_arr.npy"), 
                        np.load('trained_model/' + self.model_name + '/' + self.model_name +"_valid_loss_arr.npy"))