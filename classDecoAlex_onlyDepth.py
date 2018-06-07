from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import numpy as np
# import matplotlib.pyplot as plt
from torch.nn.modules.module import _addindent
import Alex


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
        # get_output_shape(model,module)
        # if hasattr(module,'output'):
        #    tmpstr += ' output_shape={}'.format(module.output.size())
        #    tmpstr += '\n'   
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

    def __init__(self):
        super(DECO, self).__init__()
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
        self.res1 = ResidualBlock(64)
        self.res2 = ResidualBlock(64)
        self.res3 = ResidualBlock(64)
        self.res4 = ResidualBlock(64)
        self.res5 = ResidualBlock(64)
        self.res6 = ResidualBlock(64)
        self.res7 = ResidualBlock(64)
        self.res8 = ResidualBlock(64)
        # convoluzione2
        self.conv2 = nn.Conv2d(64, 3, 1, stride=1)  # 1 canale, 3 kernels,
        # deconvolution-upsampling porta a 3x228x228
        self.deconv = nn.ConvTranspose2d(3, 3, 8, stride=4, padding=2, groups=3, bias=False)

    def forward(self, x):
        #        import code
        #        code.interact(local=locals())
        #	import ipdb; ipdb.set_trace()
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
        # import code
        # code.interact(local=locals())

        return x


class DecoAlexNet(nn.Module):
    def __init__(self, num_classes):
        super(DecoAlexNet, self).__init__()
        self.Deco = DECO()
        self.Alex = Alex.alexnet(pretrained=True)
        num_feats = self.Alex.classifier[6].in_features
        class_model = list(self.Alex.classifier.children())
        class_model.pop()
        class_model.append(nn.Linear(num_feats, num_classes))
        self.Alex.classifier = nn.Sequential(*class_model)

    def forward(self, x):
        x = self.Deco(x)
        #	import code
        #	code.interact(local=locals())
        #        vision.utils.save_image(x, './images/output_deco.jpg')
        #        sm.imsave('./images/output_deco.jpg', x)
        # import ipdb; ipdb.set_trace()
        # x = Image.cpu().numpy()
        # import code
        # code.interact(local=locals())
        x = self.Alex(x)
        return x
