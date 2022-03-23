import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.transforms import ToTensor

# import model
from model.VESPCN import VESPCN
# from model.init_weight import init_weights

# import dataset
from datasetProcess.vimeo90k import vimeo90k

class Trainer_VESPCN:
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
        if(self.model_name == 'VESPCN'):
            self.model = VESPCN(n_colors=args.n_colors, scale=args.scale, n_sequence=args.n_sequence, device=self.device).to(self.device)
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

    def train(self):
        # init_weights(self.model)

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
                inputs = data_batch[0]
                inputs = inputs.to(self.device)
                labels = data_batch[1]
                labels = labels.to(self.device)

                # 一次梯度下降
                self.optimizer.zero_grad()
                outputs, loss_mc_mse, loss_mc_huber = self.model(inputs)
                loss_espcn = self.criterion(outputs, labels)
                loss_mc = self.args.beta * loss_mc_mse + self.args.lamda * loss_mc_huber
                loss = loss_espcn + loss_mc
                loss.backward()
                self.optimizer.step()

                # train_loss
                running_loss += loss.item()
                if i_batch % 10 == 9:
                    # train_loss
                    if self.test == True:
                        print('[%d, %5d] trainLoss:%.3f' %
                                (epoch + 1, i_batch + 1, running_loss/10))
                    train_loss_arr = np.append(train_loss_arr,running_loss/10)
                    running_loss = 0.0

                    # valid_loss
                    validDataIter = iter(self.validDataLoader)
                    data_valid = validDataIter.next()
                    inputs_valid = data_valid[0]
                    inputs_valid = inputs_valid.to(self.device)
                    labels_valid = data_valid[1]
                    labels_valid = labels_valid.to(self.device)
                    outputs_valid, loss_mc_mse_valid, loss_mc_huber_valid = self.model(inputs_valid)
                    loss_espcn_valid = self.criterion(outputs_valid, labels_valid)
                    loss_mc_valid = self.args.beta * loss_mc_mse_valid + self.args.lamda * loss_mc_huber_valid
                    loss_valid = loss_espcn_valid + loss_mc_valid
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
