import torch
from torchsummary import summary
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def model_summary(model_name):
    device = torch.device('cuda:0')
    if(model_name == 'ESPCN'):
        from ESPCN import ESPCN
        net = ESPCN(n_colors=3, scale=2)
        net.to(device)
        print(summary(net, (3, 240, 360),device='cuda'))

    elif(model_name == 'ESPCN_modified'):
        from ESPCN_modified import ESPCN_modified
        net = ESPCN_modified(n_colors=3, scale=2)
        net.to(device)
        print(summary(net, (3, 240, 360),device='cuda'))

    elif(model_name == 'ESPCN_multiframe'):
        from ESPCN_multiframe import ESPCN_multiframe
        net = ESPCN_multiframe(n_colors=3, scale=2, n_sequence=3)
        net.to(device)
        print(summary(net, (3, 3, 240, 360),device='cuda'))
    elif(model_name == 'ESPCN_multiframe2'):
        from ESPCN_multiframe2 import ESPCN_multiframe2
        net = ESPCN_multiframe2(n_colors=3, scale=2, n_sequence=3)
        net.to(device)
        print(summary(net, (3, 3, 240, 360),device='cuda'))
    elif(model_name == 'motioncompensator'):
        from motioncompensator import MotionCompensator
        net = MotionCompensator(n_colors=3, device=device)
        net.to(device)
        # print(summary(net, (3, 240, 360),device='cuda'))
        frame1 = torch.ones(1,3,240,360).to(device)
        frame2 = torch.ones_like(frame1).to(device)
        frame_out = net(frame1,frame2)
        print(frame_out[0].shape, frame_out[1].shape)
    else:
        pass

if __name__ =="__main__":
    # model_summary("ESPCN")
    # model_summary("ESPCN_modified")
    # model_summary("ESPCN_multiframe")
    # model_summary("ESPCN_multiframe2")
    model_summary("motioncompensator")