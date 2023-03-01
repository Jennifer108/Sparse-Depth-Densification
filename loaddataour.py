import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import random
from nyu_transform import *



class depthDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, transform=None):
        self.frame = np.loadtxt(csv_file)
        self.transform = transform

    def __getitem__(self, idx):
        image_name = "E:\\semanticSegmentation\\DepthEstimation\\nyu\\nyu_images\\"+str(self.frame[idx]-1)[:-2]+".jpg"

        errormap_name="E:\\semanticSegmentation\\DepthEstimation\\nyu\\nyu_smcdepths\\12error\\"+str(self.frame[idx]-1)[:-2]+".png"
        
        smcdepth_name="E:\\semanticSegmentation\\DepthEstimation\\nyu\\nyu_smcdepths\\12\\"+str(self.frame[idx]-1)[:-2]+".png" #smcdepth
        # smcdepth_name="E:\\semanticSegmentation\\DepthEstimation\\nyu\\nyu_smclabel\\"+str(self.frame[idx]-1)[:-2]+".png"   #smclabel
        # smcdepth_name="E:\\semanticSegmentation\\DepthEstimation\\nyu\\nyu_depths\\"+str(self.frame[idx]-1)[:-2]+".png"   #真实深度

        depth_name = "E:\\semanticSegmentation\\DepthEstimation\\nyu\\nyu_depths\\"+str(self.frame[idx]-1)[:-2]+".png"


        edge_name="E:\\semanticSegmentation\\DepthEstimation\\nyu\\sobel\\"+str(self.frame[idx]-1)[:-2]+".png"


        image = Image.open(image_name)
        smcdepth=Image.open(smcdepth_name)
        edge=Image.open(edge_name)
        depth = Image.open(depth_name)
        errormap=Image.open(errormap_name)
        
        sample = {'image': image,'smcdepth':smcdepth,'errormap':errormap,'edge':edge, 'depth': depth}
        sample = self.transform(sample)
        # if self.transform:
        #     sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.frame)


class depthDataset1(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, transform=None):
        self.frame = pd.read_csv(csv_file, header=None)
        self.transform = transform

    def __getitem__(self, idx):
        image_name = self.frame.iloc[idx, 0]
        depth_name = self.frame.iloc[idx, 1]

        image = Image.open(image_name)
        depth = Image.open(depth_name)
        # print(image_name[-16:-11])

        # smcdepth=Image.open('data/nyu2_test_smclabel/'+image_name[-16:-11]+'.png')   #smclabel
        # smcdepth = Image.open(depth_name)    #真实深度
        smcdepth=Image.open('data/nyuv2_test_smcdepth/12/'+image_name[-16:-11]+'.png') #合成的smc深度


        errormap=Image.open('data/nyuv2_test_smcdepth/12errrornyuv2test/'+image_name[-16:-11]+'.png') #合成的smc深度
        edge=Image.open('data/nyutestsobel/'+image_name[-16:-11]+'.png') #合成的smc深度

        sample = {'image': image,'smcdepth':smcdepth,'errormap':errormap,'edge':edge, 'depth': depth}

        if self.transform:
            sample = self.transform(sample)
        xx = {'name':image_name[-16:-11]+'.png'}
        sample.update(xx)
        return sample

    def __len__(self):
        return len(self.frame)


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
