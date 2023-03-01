from collections import OrderedDict
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils import model_zoo
import copy
import numpy as np
from . import modulesour
from torchvision import utils

from . import senet
from . import resnet
from . import densenet

class model(nn.Module):
    def __init__(self, Encoder, num_features, block_channel):

        super(model, self).__init__()

        self.E = Encoder
        self.D = modulesour.D(num_features)
        self.MFF = modulesour.MFF(block_channel)
        self.R = modulesour.R(block_channel)


    def forward(self, x,smcdepth,errormap,edge):
        x_block1, x_block2, x_block3, x_block4 = self.E(x,smcdepth,errormap)
        x_decoder = self.D(x_block1, x_block2, x_block3, x_block4)
        x_mff = self.MFF(x_block1, x_block2, x_block3, x_block4,[x_decoder.size(2),x_decoder.size(3)])
      
        consum=torch.cat((x_decoder, x_mff), 1)
        
        # edge=edge[:, :, 0::2, 0::2]
        # edge=edge.float()

        # smcdepth = smcdepth[:, :, 0::2, 0::2]
        # smcdepth=smcdepth.float()

        # out = self.R(torch.cat((consum, edge),1))
    
        out = self.R(torch.cat((x_decoder, x_mff), 1))
        return out
