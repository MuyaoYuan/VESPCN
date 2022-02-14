import numpy as np
from PIL import Image

def pictureProcess(picture_tensor):
    picture_array = picture_tensor.cpu().detach().numpy()
    picture_array = picture_array.transpose((0,2,3,1))
    im_list = []
    for picture_item in picture_array:
        im_list.append(Image.fromarray(np.uint8(picture_item*255)))
    return im_list

def framesProcess(frames_tensor):
    frames_array = frames_tensor.cpu().detach().numpy()
    frames_array = frames_array.transpose((0,1,3,4,2))
    frames_list = []
    for frames_item in frames_array:
        frames = []
        for frame in frames_item:
            frames.append(Image.fromarray(np.uint8(frame*255)))
        frames_list.append(frames)
    return frames_list