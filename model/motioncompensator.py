import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MotionCompensator(nn.Module):
    def __init__(self, n_colors, device):
        super(MotionCompensator, self).__init__()

        print("Creating Motion Compensator")

        self.device = device
                        
        # Coarse flow
        coarse_flow = [nn.Conv2d(2*n_colors, 24, kernel_size=5, stride=2, padding=2), nn.ReLU(inplace=True)]
        # nn.Conv2d(2*3, 24*3, kernel_size=5, groups=3, stride=2)
        coarse_flow.extend([nn.Conv2d(24, 24, kernel_size=3, padding=1), nn.ReLU(True)])
        coarse_flow.extend([nn.Conv2d(24, 24, kernel_size=5, stride=2, padding=2), nn.ReLU(True)])
        coarse_flow.extend([nn.Conv2d(24, 24, kernel_size=3, padding=1), nn.ReLU(True)])
        coarse_flow.extend([nn.Conv2d(24, 32, kernel_size=3, padding=1), nn.Tanh()])
        coarse_flow.extend([nn.PixelShuffle(4)])

        self.C_flow = nn.Sequential(*coarse_flow)

        # Fine flow
        fine_flow = [nn.Conv2d(3*n_colors+2, 24, kernel_size=5, stride=2, padding=2), nn.ReLU(inplace=True)]
        for _ in range(3):
            fine_flow.extend([nn.Conv2d(24, 24, kernel_size=3, padding=1), nn.ReLU(True)])
        fine_flow.extend([nn.Conv2d(24, 8, kernel_size=3, padding=1), nn.Tanh()])
        fine_flow.extend([nn.PixelShuffle(2)])

        self.F_flow = nn.Sequential(*fine_flow)

    def forward(self, frame_1, frame_2):
        # Create identity flow
        x = np.linspace(-1, 1, frame_1.shape[3])
        y = np.linspace(-1, 1, frame_1.shape[2])
        xv, yv = np.meshgrid(x, y)
        id_flow = np.expand_dims(np.stack([xv, yv], axis=-1), axis=0)
        self.identity_flow = torch.from_numpy(id_flow).float().to(self.device)

        # Coarse flow
        coarse_in = torch.cat((frame_1, frame_2), dim=1)
        coarse_out = self.C_flow(coarse_in)
        frame_2_compensated_coarse = self.warp(frame_2, coarse_out)
        
        # Fine flow
        fine_in = torch.cat((frame_1, frame_2, frame_2_compensated_coarse, coarse_out), dim=1)
        fine_out = self.F_flow(fine_in)
        flow = (coarse_out + fine_out)

        frame_2_compensated = self.warp(frame_2, flow)

        return frame_2_compensated, flow

    def warp(self, img, flow):
        # https://discuss.pytorch.org/t/solved-how-to-do-the-interpolating-of-optical-flow/5019
        # permute flow N C H W -> N H W C
        # torch.clamp(input, min, max, out=None)
        # 将输入input张量每个元素的范围限制到区间 [min,max]，返回结果到一个新张量。   
        img_compensated = F.grid_sample(img, (-flow.permute(0,2,3,1)+self.identity_flow).clamp(-1,1), padding_mode='border')
        return img_compensated