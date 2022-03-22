import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.transforms import ToTensor

# import model
from model.motioncompensator import MotionCompensator
from model.init_weight import init_weights

# import loss function
from approx_huber_loss import Approx_Huber_Loss

# import dataset
from datasetProcess.vimeo90k import vimeo90k

class Trainer_MC:
    def __init__(self, args):
        self.args = args
        if(args.task == 'preparation'):
            self.test = True
        else:
            self.test = False
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.epochs = args.epochs
        # model select
        self.model_name = args.model
        if(self.model_name == 'MC'):
            self.model = MotionCompensator(n_colors=args.n_colors, device=self.device).to(self.device)
        else:
            print('Please Enter Appropriate Model!!!')

        self.lr = args.lr
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        self.criterion = nn.MSELoss()

        # Dateset select
        if(args.dataset_name == 'vimeo90k'):
            self.dataset_path = args.dataset_path
            self.batch_size = args.batch_size
            # print(args.num_workers,args.pin_memory)
            self.trainDataset = vimeo90k(path = self.dataset_path)
            self.trainDataLoader = DataLoader(self.trainDataset, batch_size=self.batch_size, shuffle=True,num_workers=args.num_workers,pin_memory=args.pin_memory)
            self.validDataset = vimeo90k(path = self.dataset_path, train=False)
            self.validDataLoader = DataLoader(self.validDataset, batch_size=self.batch_size, shuffle=True,num_workers=args.num_workers,pin_memory=args.pin_memory)
        else:
            print('Please Enter Appropriate Dataset!!!')

        # loss function
        self.flow_loss = Approx_Huber_Loss(device=self.device)

    def train(self):
        init_weights(self.model)

        running_loss = 0.0
        valid_loss = 0.0
        train_loss_arr = np.array([])
        valid_loss_arr = np.array([])

        for epoch in range(self.epochs):
            print(epoch)
            if(epoch %10 == 9 and self.lr - 1e-6 > 1e-6): 
                self.lr = self.lr/10
                self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
                print('Changing Learing Rate To: ' + str(self.lr))
            for i_batch, data_batch in enumerate(self.trainDataLoader):
                # 数据
                frames = data_batch[0]
                frame1 = frames[:,1]
                frame1 = frame1.to(self.device)
                frame2 = frames[:,0]
                frame2 = frame2.to(self.device)
                frame3 = frames[:,2]
                frame3 = frame3.to(self.device)
                
                # 第一次梯度下降
                frame2_compensated, flow = self.model(frame1, frame2)
                self.optimizer.zero_grad()
                loss = self.criterion(frame2_compensated, frame1) + self.args.lamda * self.flow_loss(flow)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                # 第二次梯度下降
                frame2_compensated, flow = self.model(frame1, frame3)
                self.optimizer.zero_grad()
                loss = self.criterion(frame2_compensated, frame1) + self.args.lamda * self.flow_loss(flow)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

                # train_loss
                if i_batch % 10 == 9:
                    # train_loss
                    if self.test == True:
                        print('[%d, %5d] trainLoss:%.3f' %
                                (epoch + 1, i_batch + 1, running_loss/20))
                    train_loss_arr = np.append(train_loss_arr,running_loss/20)
                    running_loss = 0.0

                    # valid_loss
                    validDataIter = iter(self.validDataLoader)
                    data_valid = validDataIter.next()
                    frames_valid = data_valid[0]
                    frame1_valid = frames_valid[:,1]
                    frame1_valid = frame1_valid.to(self.device)
                    frame2_valid = frames_valid[:,0]
                    frame2_valid = frame2_valid.to(self.device)
                    frame2_compensated_valid, flow_valid = self.model(frame1_valid, frame2_valid)
                    loss_valid = self.criterion(frame2_compensated_valid, frame1_valid) + self.args.lamda * self.flow_loss(flow_valid)
                    valid_loss += loss_valid.item()
                    if self.test == True:
                        print('[%d, %5d] validLoss:%.3f' %
                                (epoch + 1, i_batch + 1, valid_loss))
                    valid_loss_arr = np.append(valid_loss_arr,valid_loss)
                    valid_loss = 0.0
                if self.test == True:
                    if i_batch >= 19:
                        break
            if self.test == True:
                if epoch >= 21:
                    break
                # break

        print("Finished Training")
        if self.test == True:
            torch.save(self.model.state_dict(), 'trained_model/' + self.model_name + '/'+self.model_name + '_demo.pkl')
            np.save('trained_model/' + self.model_name + '/' + self.model_name +"_train_loss_arr_demo.npy",train_loss_arr)
            np.save('trained_model/' + self.model_name + '/' + self.model_name + "_valid_loss_arr_demo.npy",valid_loss_arr)
        else:
            torch.save(self.model.state_dict(), 'trained_model/' + self.model_name + '/' + self.model_name + '.pkl')
            np.save('trained_model/' + self.model_name + '/' + self.model_name +"_train_loss_arr.npy",train_loss_arr)
            np.save('trained_model/' + self.model_name + '/' + self.model_name + "_valid_loss_arr.npy",valid_loss_arr)
        

            


