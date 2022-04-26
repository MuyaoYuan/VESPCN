from PIL import Image
import os

if __name__ =="__main__":
    # path = "dataset/DIV2K_train_LR_bicubic_X2"
    # save_path = "dataset/DIV2K_train_HR_bicubic"
    path = "dataset/DIV2K_valid_LR_bicubic_X2"
    save_path = "dataset/DIV2K_valid_HR_bicubic"
    os.makedirs(save_path, exist_ok=True)
    upsample = 2
    file_list = os.listdir(path)
    for index in range(len(file_list)):
        # print(index)
        img = Image.open(os.path.join(path, file_list[index]))
        img_w, img_h = img.size
        img = img.resize([img_w*2, img_h*2], Image.BICUBIC)
        img.save(os.path.join(save_path, file_list[index]))
        # break
    print('data process finished')