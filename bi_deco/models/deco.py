from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.nn.modules.module import _addindent


def torch_summarize(model, show_weights=True, show_parameters=True):
    """Summarizes torch model by showing trainable parameters and weights."""
    tmpstr = model.__class__.__name__ + ' (\n'
    total_params = 0
    for key, module in model._modules.items():
        # if it contains layers let call it recursively to get params and weights
        if type(module) in [
            torch.nn.modules.container.Container,
            torch.nn.modules.container.Sequential
        ]:
            modstr = torch_summarize(module)
        else:
            modstr = module.__repr__()
        modstr = _addindent(modstr, 2)

        params = sum([np.prod(p.size()) for p in module.parameters()])
        weights = tuple([tuple(p.size()) for p in module.parameters()])
        total_params += params
        tmpstr += '  (' + key + '): ' + modstr
        if show_weights:
            tmpstr += ', weights={}'.format(weights)
        if show_parameters:
            tmpstr += ', parameters={}'.format(params)
        tmpstr += '\n'
    tmpstr = tmpstr + ')'
    tmpstr = tmpstr + '\n total number of parameters={}'.format(total_params)
    return tmpstr


class ResidualBlock(nn.Module):
    def __init__(self, filters, pad=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=pad, bias=True)
        self.bn1 = nn.BatchNorm2d(filters, eps=0.0001)
        self.Lrelu = nn.LeakyReLU(negative_slope=0.02)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=pad, bias=True)
        self.bn2 = nn.BatchNorm2d(filters, eps=0.0001)
        self.downsample = None
        self.filters = filters

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.Lrelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.Lrelu(out)
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


class DECO(nn.Module):
    def __init__(self, alex_net):
        super(DECO, self).__init__()
        self.alex_net = alex_net
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        # convoluzione1
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

        # convoluzione2
        # self.conv2 = nn.Conv2d(64, 3, 1, stride=1)  # 1 canale, 3 kernels,

        # deconvolution-upsampling porta a 3x228x228

        # TODO: we changed padding from 3 to 0
        # self.deconv = nn.ConvTranspose2d(3, 3, 8, stride=4, padding=0, groups=3, bias=False)
        if alex_net:
            # convoluzione2
            self.conv2 = nn.Conv2d(64, 3, 1, stride=1)  # 1 canale, 3 kernels,
            # deconvolution-upsampling porta a 3x228x228
            # TODO: we changed padding from 3 to 0
            self.deconv = nn.ConvTranspose2d(3, 3, 8, stride=4, padding=0, groups=3, bias=False)
            self.last_layer = self.deconv
        else:
            self.last_pool = nn.MaxPool2d(3, stride=2)
            # TODO: dropout here
            self.last_layer = nn.Linear(64 * 27 * 27, 2500 * 3)

    def forward(self, x):
        x = x.view(x.size(0), 1, 224, 224)
        # print("x = x.view(x.size(0), 1, 224, 224)", x.size())
        x = self.conv1(x)
        # print("x = self.conv1(x)", x.size())
        x = self.bn1(x)
        # print("x = self.bn1(x)", x.size())
        x = self.Lrelu(x)
        # print("x = self.Lrelu(x)", x.size())
        x = self.pool(x)
        # print("x = self.pool(x)", x.size())
        x = self.res1(x)
        # print("x = self.res1(x)", x.size())
        x = self.res2(x)
        # print("x = self.res2(x)", x.size())
        x = self.res3(x)
        # print("x = self.res3(x)", x.size())
        x = self.res4(x)
        # print("x = self.res4(x)", x.size())
        x = self.res5(x)
        # print("x = self.res5(x)", x.size())
        x = self.res6(x)
        # print("x = self.res6(x)", x.size())
        x = self.res7(x)
        # print("x = self.res7(x)", x.size())
        x = self.res8(x)
        # print("x = self.res8(x)", x.size())

        if self.alex_net:
            x = self.conv2(x)
            # print("x = self.conv2(x)", x.size())
            # x = self.deconv(x)
        else:
            x = self.last_pool(x)
            x = x.view(x.size(0), -1)

        x = self.last_layer(x)
        # print("x = self.last_layer(x)", x.size())
        return x

# class DecoAlexNet(nn.Module):
#     def __init__(self, num_classes):
#         super(DecoAlexNet, self).__init__()
#         self.Deco = DECO()
#         self.Alex = Alex.alexnet(pretrained=True)
#         num_feats = self.Alex.classifier[6].in_features
#         class_model = list(self.Alex.classifier.children())
#         class_model.pop()
#         class_model.append(nn.Linear(num_feats, num_classes))
#         self.Alex.classifier = nn.Sequential(*class_model)
#
#     def forward(self, x):
#         x = self.Deco(x)
#         x = self.Alex(x)
#        return x
