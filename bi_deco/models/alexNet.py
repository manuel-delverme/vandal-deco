from __future__ import print_function
import pointnet
import torch.nn as nn
import numpy as np
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import torch
import torch.nn.functional as F


class AlexNet(nn.Module):
    def __init__(self, drop=0.0):
        super(AlexNet, self).__init__()
        print("LOL")
