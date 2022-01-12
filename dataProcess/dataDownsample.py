from PIL import Image
import os

if __name__ =="__main__":
    path = "dataset/vimeo90k/vimeo_triplet/sequences"
    save_path = "dataset/vimeo90k/vimeo_triplet/sequences_LR_bicubic_X2"
    downsample = 2
    file_list_1 = os.listdir(path)
    for index1 in range(len(file_list_1)):
        print(index1)
        file_list_2 = os.listdir(os.path.join(path, file_list_1[index1]))
        for index2 in range(len(file_list_2)):
            file_list_3 = os.listdir(os.path.join(path, file_list_1[index1], file_list_2[index2]))
            if(not os.path.exists(os.path.join(save_path, file_list_1[index1], file_list_2[index2]))):
                os.makedirs(os.path.join(save_path, file_list_1[index1], file_list_2[index2]))
            for index3 in range(len(file_list_3)):
                img = Image.open(os.path.join(path, file_list_1[index1], file_list_2[index2], file_list_3[index3]))
                img_w, img_h = img.size
                img.thumbnail((img_w/downsample, img_h/downsample),Image.BICUBIC)
                img.save(os.path.join(save_path, file_list_1[index1], file_list_2[index2], file_list_3[index3]))
        # if index1 == 1:
        #     break;
    print('data process finished')