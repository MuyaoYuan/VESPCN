from PIL import Image
import os

if __name__ =="__main__":
    path = "dataset/walk"
    save_path = "dataset/walk_downscale"
    downsample = 2
    file_list = os.listdir(path)
    for index in range(len(file_list)):
        img = Image.open(os.path.join(path, file_list[index]))
        img_w, img_h = img.size
        img.thumbnail((img_w/downsample, img_h/downsample))
        img.save(os.path.join(save_path,'down'+file_list[index]))