import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import random
from nyu_transform import *
import os
import matplotlib.pyplot as plt


class depthDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, transform=None):
        self.frame = np.loadtxt(csv_file)
        self.transform = transform
        self.dd="E:\\semanticSegmentation\\DepthEstimation\\make3d\\Img\\Train400Img\\"
        self.data_list = os.listdir(self.dd)

    def __getitem__(self, idx):
        image_name = self.dd+str(self.data_list[idx])
        depth_name = "E:\\semanticSegmentation\\DepthEstimation\\make3d\\depth\\Train400DepthMap\\"+'depth_sph_corr'+str(self.data_list[idx])[3:-4]+'.png'

        errormap_name="E:\\semanticSegmentation\\DepthEstimation\\make3d\\our\\12error\\"+str(self.data_list[idx])[:-4]+".png" 
        smcdepth_name="E:\\semanticSegmentation\\DepthEstimation\\make3d\\sparsedepth\\1000sp\\"+str(self.data_list[idx])[:-4]+".png"

        edge_name="E:\\semanticSegmentation\\DepthEstimation\\make3d\\our\\5\\"+str(self.data_list[idx])[:-4]+".png"


        image = Image.open(image_name)
        smcdepth=Image.open(smcdepth_name)
        edge=Image.open(edge_name)
        depth = Image.open(depth_name)
        errormap=Image.open(errormap_name)
        
        image = image.resize((640,480))
        smcdepth = smcdepth.resize((640,480))
        edge = edge.resize((640,480))
        depth = depth.resize((640,480))
        errormap = errormap.resize((640,480))
        
        sample = {'image': image,'smcdepth':smcdepth,'errormap':errormap,'edge':edge, 'depth': depth}
        sample = self.transform(sample)
        # if self.transform:
        #     sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.data_list)


class depthDataset1(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, transform=None):
        self.frame = pd.read_csv(csv_file, header=None)
        self.transform = transform
        self.dd="E:\\semanticSegmentation\\DepthEstimation\\make3d\\Img\\Test134\\"
        self.data_list = os.listdir(self.dd)

    def __getitem__(self, idx):


        image_name = self.dd+str(self.data_list[idx])
        depth_name = "E:\\semanticSegmentation\\DepthEstimation\\make3d\\depth\\Test134DepthMap\\"+'depth_sph_corr'+str(self.data_list[idx])[3:-4]+'.png'

        errormap_name="E:\\semanticSegmentation\\DepthEstimation\\make3d\\our\\12errortest\\"+str(self.data_list[idx])[:-4]+".png" 
        smcdepth_name="E:\\semanticSegmentation\\DepthEstimation\\make3d\\sparsedepth\\1000sptest\\"+str(self.data_list[idx])[:-4]+".png"

        edge_name="E:\\semanticSegmentation\\DepthEstimation\\make3d\\our\\5test\\"+str(self.data_list[idx])[:-4]+".png"


        image = Image.open(image_name)
        smcdepth=Image.open(smcdepth_name)
        edge=Image.open(edge_name)
        depth = Image.open(depth_name)
        errormap=Image.open(errormap_name)

        image = image.resize((640,480))
        smcdepth = smcdepth.resize((640,480))
        edge = edge.resize((640,480))
        depth = depth.resize((640,480))
        errormap = errormap.resize((640,480))

        
        # plt.imshow(depth)
        # plt.show()



        sample = {'image': image,'smcdepth':smcdepth,'errormap':errormap,'edge':edge, 'depth': depth}

        if self.transform:
            sample = self.transform(sample)
        xx = {'name':str(self.data_list[idx])[:-4]+'.png'}
        sample.update(xx)
        return sample

    def __len__(self):
        return len(self.data_list)


def getTrainingData(batch_size=64):
    __imagenet_pca = {
        'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
        'eigvec': torch.Tensor([
            [-0.5675,  0.7192,  0.4009],
            [-0.5808, -0.0045, -0.8140],
            [-0.5836, -0.6948,  0.4203],
        ])
    }
    __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}

    transformed_training = depthDataset(csv_file='./data/nyu2_train.txt',
                                        transform=transforms.Compose([
                                            Scale(240),
                                            RandomHorizontalFlip(),#按照一定的概率翻转
                                            RandomRotate(5),  #按照一定概率旋转
                                            CenterCrop([304, 228],[152, 114]),  #],[304, 228],[152, 114]
                                            ToTensor(),
                                            Lighting(0.1, __imagenet_pca[
                                                'eigval'], __imagenet_pca['eigvec']),
                                            ColorJitter(   #亮度对比度变化
                                                brightness=0.4,
                                                contrast=0.4,
                                                saturation=0.4,
                                            ),
                                            Normalize(__imagenet_stats['mean'],
                                                      __imagenet_stats['std'])
                                        ]))

    dataloader_training = DataLoader(transformed_training, batch_size,
                                     shuffle=True, num_workers=4, pin_memory=False)

    return dataloader_training


def getTestingData(batch_size=64):
    __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}
    # scale = random.uniform(1, 1.5)
    transformed_testing = depthDataset1(csv_file='./data/nyu2_test.csv',
                                       transform=transforms.Compose([
                                           Scale(240),
                                           CenterCrop([304, 228], [304, 228]),
                                           ToTensor(is_test=True),
                                           Normalize(__imagenet_stats['mean'],
                                                     __imagenet_stats['std'])
                                       ]))

    dataloader_testing = DataLoader(transformed_testing, batch_size,
                                    shuffle=False, num_workers=0, pin_memory=False)

    return dataloader_testing
