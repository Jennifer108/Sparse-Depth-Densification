import argparse
import torch
import torch.nn.parallel

from models import modules, net, resnet, densenet, senet
import numpy as np
import loaddata_demo as loaddata
import pdb

import matplotlib.image
import matplotlib.pyplot as plt
import os
plt.set_cmap("jet")


def define_model(is_resnet, is_densenet, is_senet):
    if is_resnet:
        original_model = resnet.resnet50(pretrained = True)
        Encoder = modules.E_resnet(original_model) 
        model = net.model(Encoder, num_features=2048, block_channel = [256, 512, 1024, 2048])
    if is_densenet:
        original_model = densenet.densenet161(pretrained=True)
        Encoder = modules.E_densenet(original_model)
        model = net.model(Encoder, num_features=2208, block_channel = [192, 384, 1056, 2208])
    if is_senet:
        original_model = senet.senet154(pretrained='imagenet')
        Encoder = modules.E_senet(original_model)
        model = net.model(Encoder, num_features=2048, block_channel = [256, 512, 1024, 2048])

    return model
   

def main():
    model = define_model(is_resnet=True, is_densenet=False, is_senet=False)
    # model = torch.nn.DataParallel(model).cuda()
    # model.load_state_dict(torch.load('checkpoint15.pth'))

    model =model.cuda()
    model.load_state_dict({k.replace('module.',''):v for k,v in torch.load("checkpoint19.pth").items()})
    model.eval()

    testPath="E:\\semanticSegmentation\\DepthEstimation\\ourmethod\\net\\data\\nyu2_test\\"

    for line in os.listdir(testPath):
        print(line)
        d=line[-10:]
        if line[-10:]=="colors.png":
            nyu2_loader = loaddata.readNyu2(testPath+line)
            test(nyu2_loader, model,line)


def test(nyu2_loader, model,line):
    for i, image in enumerate(nyu2_loader):     
        image = torch.autograd.Variable(image, volatile=True).cuda()
        out = model(image)
        matplotlib.image.imsave('E:\\semanticSegmentation\\DepthEstimation\\ourmethod\\net\\data\\results\\smc\\'+line, out.view(out.size(2),out.size(3)).data.cpu().numpy())

if __name__ == '__main__':
    main()
