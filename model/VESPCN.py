from matplotlib import scale
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.motioncompensator import MotionCompensator
from model.ESPCN_multiframe2 import ESPCN_multiframe2
from approx_huber_loss import Approx_Huber_Loss

class VESPCN(nn.Module):
    def __init__(self, n_colors, scale, n_sequence, device):
        self.name = 'VESPCN'
        self.device = device 
        super(VESPCN, self).__init__()
        print("Creating VESPCN")
        
        self.mseloss = nn.MSELoss()
        self.huberloss = Approx_Huber_Loss(device=device)
        self.motionCompensator = MotionCompensator(n_colors=n_colors, device=device).to(self.device)
        self.espcn = ESPCN_multiframe2(n_colors=n_colors, scale=scale, n_sequence=n_sequence).to(self.device)
        
        # load model
        self.motionCompensator.load_state_dict(torch.load('trained_model/ESPCN_multiframe2/ESPCN_multiframe2.pkl'), strict=False)
        self.espcn.load_state_dict(torch.load('trained_model/MC/MC.pkl'), strict=False)
        
    def forward(self, frames):
        # frames [N, n_sequence, n_colors, H, W]
        
        frame1 = frames[:,0]
        frame2 = frames[:,1]
        frame3 = frames[:,2]

        frame1_compensated, flow1 = self.motionCompensator(frame2, frame1)
        frame3_compensated, flow2 = self.motionCompensator(frame2, frame3)
        
        loss_mc_mse = self.mseloss(frame1_compensated, frame2) + self.mseloss(frame3_compensated, frame2)
        loss_mc_huber = self.huberloss(flow1) + self.huberloss(flow2)
        
        frames_compensated = torch.stack((frame1_compensated, frame2, frame3_compensated), dim = 1)

        return self.espcn(frames_compensated), loss_mc_mse, loss_mc_huber