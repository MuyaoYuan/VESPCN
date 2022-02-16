from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.transforms import ToTensor

from imageProcess import *

def dataset_test(dataset_name):
    if(dataset_name == 'DIV2K'):
        from DIV2K import DIV2K
        path_in = "dataset/DIV2K_train_LR_bicubic_X2"
        path_label = "dataset/DIV2K_train_HR"
        batch_size = 1
        myDataset = DIV2K(dir_input=path_in,dir_label=path_label,transform=ToTensor())
        myDataLoader = DataLoader(myDataset, batch_size=batch_size, shuffle=True)
        dataIter = iter(myDataLoader)
        dataItem = dataIter.next()

        # print(img_in.shape, img_label.shape, title_in, title_label)
        print(dataItem[0].shape, dataItem[1].shape, dataItem[2], dataItem[3])

    elif(dataset_name == 'vimeo90k'):
        from vimeo90k import vimeo90k
        path = "dataset/vimeo90k/vimeo_triplet"
        batch_size = 1
        myDataset = vimeo90k(path = path, train=True)
        myDataLoader = DataLoader(myDataset, batch_size=batch_size, shuffle=True)

        dataIter = iter(myDataLoader)
        dataItem = dataIter.next()
        
        print(len(myDataset))
        print(dataItem[0].shape, dataItem[1].shape, dataItem[2], dataItem[3], dataItem[4], dataItem[5])

        inputs = framesProcess(dataItem[0])
        # labels = framesProcess(dataItem[1])
        inputs[0][0].save("input_0.png")
        inputs[0][1].save("input_1.png")
        inputs[0][2].save("input_2.png")
        # labels[0][0].save("label_0.png")
        # labels[0][1].save("label_1.png")
        # labels[0][2].save("label_2.png")
        labels = pictureProcess(dataItem[1])
        labels[0].save("label.png")
    else:
        pass

if __name__ =="__main__":
    # dataset_test('DIV2K')
    # print('\n')
    dataset_test('vimeo90k')