from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable

import pointnet


class DECO_medium_conv(nn.Module):
    def __init__(self, drop=0.0, softmax=True):
        super(DECO_medium_conv, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3)
        # BN e Leacky ReLU
        self.bn1 = nn.BatchNorm1d(64)
        self.Lrelu = nn.LeakyReLU(negative_slope=0.01)
        # maxPooling
        self.pool = nn.MaxPool2d(3, stride=2)  # 64x57x57
        # 8 blocchi di residual
        self.res1 = SEBasicBlock(64, 64)
        self.res2 = SEBasicBlock(64, 64)
        self.res3 = SEBasicBlock(64, 64)
        self.res4 = SEBasicBlock(64, 64)
        self.res5 = SEBasicBlock(64, 64)
        self.res6 = SEBasicBlock(64, 64)
        self.res7 = SEBasicBlock(64, 64)
        self.res8 = SEBasicBlock(64, 64)

        # TODO: pointNet.fc3 = torch.nn.Linear(pointNet.fc3.in_features, 51)
        # self.classifier = pointnet.PointNetClassifier(k=16)
        self.drop = drop
        self.softmax = softmax
        # self.feat = PointNetfeat(num_points = 2500, global_feat = True)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.rb1 = ResidualBlock(64)
        self.rb2 = ResidualBlock(64)
        self.maxpool1 = nn.MaxPool2d(3, stride=2)
        self.resize_filters1 = resize_filters(64, 128)
        self.rb3 = ResidualBlock(128)
        self.rb4 = ResidualBlock(128)
        self.maxpool2 = nn.MaxPool2d(3, stride=2)
        self.resize_filters2 = resize_filters(128, 256)
        self.rb5 = ResidualBlock(256)
        self.rb6 = ResidualBlock(256)
        self.maxpool3 = nn.MaxPool2d(3, stride=2)
        self.resize_filters3 = resize_filters(256, 512)
        self.rb7 = ResidualBlock(512)
        self.rb8 = ResidualBlock(512)
        self.maxpool4 = nn.MaxPool2d(3, stride=2)
        # TODO: stolen from Chira
        # convoluzione2
        self.conv2 = nn.Conv2d(64, 3, 1, stride=1)  # 1 canale, 3 kernels,
        # deconvolution-upsampling porta a 3x228x228
        self.deconv = nn.ConvTranspose2d(3, 3, 8, stride=4, padding=2, groups=3, bias=False)
        # TODO: end stolen

        # self.fc1 = nn.Linear(self.n_size, 4096)
        # self.fc2 = nn.Linear(4096, 7500)

        # self.tanfc2 = nn.Hardtanh(min_val=-0.5, max_val=0.5, inplace=False, min_value=None, max_value=None)

    def forward(self, x):
        x = x.view(x.size(0), 1, 228, 228)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.Lrelu(x)
        x = self.pool(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.res6(x)
        x = self.res7(x)
        x = self.res8(x)

        x = self.conv2(x)
        x = self.deconv(x)

        # if self.softmax:
        #    # TODO WHYY?!?!?!?!?
        #    x = F.relu(x)
        # x = x.view(x.size(0), 3, 2500)
        # TODO: 50?! u srs?
        # x = x.view(x.size(0), 3, 50, 50)
        return x
        # return self.classifier(x)


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResidualBlock(nn.Module):
    def __init__(self, filters, pad=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=pad, bias=True)
        self.bn1 = nn.BatchNorm2d(filters)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=pad, bias=True)
        self.bn2 = nn.BatchNorm2d(filters)
        self.downsample = None
        self.filters = filters

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class resize_filters(nn.Module):
    def __init__(self, filters_in, filters_out):
        super(resize_filters, self).__init__()
        self.conv1 = nn.Conv2d(filters_in, filters_out, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(filters_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        return out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, reduction),
            nn.ReLU(inplace=True),
            nn.Linear(reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
