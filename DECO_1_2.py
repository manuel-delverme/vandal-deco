from __future__ import print_function

import Alex
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data


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
        return out


class DECO(nn.Module):

    def __init__(self, n_channels):
        super(DECO, self).__init__()
        # 3 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        # convoluzione1
        self.n_channels = n_channels
        self.conv1 = nn.Conv2d(self.n_channels, 64, 7, stride=2, padding=3)
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
        self.res9 = ResidualBlock(64)
        self.res10 = ResidualBlock(64)
        self.res11 = ResidualBlock(64)
        self.res12 = ResidualBlock(64)
        self.res13 = ResidualBlock(64)
        self.res14 = ResidualBlock(64)
        self.res15 = ResidualBlock(64)
        self.res16 = ResidualBlock(64)

        # convoluzione2
        self.conv2 = nn.Conv2d(64, 3, 1, stride=1)  # 1 canale, 3 kernels,
        # deconvolution-upsampling porta a 3x228x228
        self.deconv = nn.ConvTranspose2d(3, 3, 8, stride=4, padding=2, groups=3, bias=False)

    def forward(self, x):
        x = x.view(x.size(0), self.n_channels, 228, 228)
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
        x = self.res9(x)
        x = self.res10(x)
        x = self.res11(x)
        x = self.res12(x)
        x = self.res13(x)
        x = self.res14(x)
        x = self.res15(x)
        x = self.res16(x)

        x = self.conv2(x)
        x = self.deconv(x)
        return x


class DecoAlexNet(nn.Module):
    def __init__(self, n_channels, num_classes):
        super(DecoAlexNet, self).__init__()
        self.Deco = DECO(n_channels=n_channels)
        self.Alex = Alex.alexnet(pretrained=True)
        num_feats = self.Alex.classifier[6].in_features
        class_model = list(self.Alex.classifier.children())
        class_model.pop()
        class_model.append(nn.Linear(num_feats, 51))  # num_classes))  mettere 49 a num_classes se non funziona
        self.Alex.classifier = nn.Sequential(*class_model)

    def forward(self, x):
        x = self.Deco(x)
        x = self.Alex(x)
        return x

    def forward_deco(self, x):
        # import ipdb; ipdb.set_trace()
        x = self.Deco(x)
        return x


class DecoAlexFeat(nn.Module):
    def __init__(self, n_channels, num_classes, weights):
        super(DecoAlexFeat, self).__init__()
        self.DecoAlexNet = DecoAlexNet(n_channels=n_channels, num_classes=num_classes)
        self.DecoAlexNet.load_state_dict(torch.load(weights))
        num_feats = self.DecoAlexNet.Alex.classifier[6].in_features
        model = list(self.DecoAlexNet.Alex.classifier.children())
        model.pop()
        self.DecoAlexNet.Alex.classifier = nn.Sequential(*model)

    def forward(self, x):
        x = self.DecoAlexNet(x)
        return x
