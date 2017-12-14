from __future__ import print_function
import pointnet
import torch.nn as nn
import numpy as np
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import torch
import torch.nn.functional as F


class DECO(nn.Module):
    def __init__(self, drop=0.0):
        super(DECO, self).__init__()
        # self.feat = PointNetCls(num_points = 2500, k = 51)
        self.feat = cls
        self.drop = drop
        # self.feat = PointNetfeat(num_points = 2500, global_feat = True)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.rb1 = ResidualBlock(64)
        self.rb2 = ResidualBlock(64)
        self.rb4 = ResidualBlock(64)
        self.rb5 = ResidualBlock(64)
        self.rb6 = ResidualBlock(64)
        self.rb7 = ResidualBlock(64)
        self.rb8 = ResidualBlock(64)
        self.maxpool = nn.MaxPool2d(3, stride=2)
        self.n_size = self._get_conv_output((1, 224, 224))
        print("fc input size:" + str(self.n_size))
        # code.interact(local=locals())

        self.fc1 = nn.Linear(self.n_size, 4096)
        self.fc2 = nn.Linear(4096, 7500)

    def _get_conv_output(self, shape):
        bs = 1
        input = Variable(torch.rand(bs, *shape))
        output_feat = self._forward_features(input)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def _forward_features(self, x):
        x = x.view(x.size(0), 1, 224, 224)
        x = self.conv1(x)
        x = F.max_pool2d(F.relu(self.bn1(x)), 2)
        x = self.rb1(x)
        x = self.rb2(x)
        x = self.rb3(x)
        x = self.rb4(x)
        x = self.rb5(x)
        x = self.rb6(x)
        x = self.rb7(x)
        x = self.rb8(x)
        x = self.maxpool(x)
        return x

    def forward(self, x):
        x = x.view(x.size(0), 1, 224, 224)
        x = self.conv1(x)
        x = F.max_pool2d(F.relu(self.bn1(x)), 2)
        x = self.rb1(x)
        x = self.rb2(x)
        x = self.rb3(x)
        x = self.rb4(x)
        x = self.rb5(x)
        x = self.rb6(x)
        x = self.rb7(x)
        x = self.rb8(x)
        x = self.maxpool(x)
        # x = self.deconv(x)
        # x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.size(0), self.n_size)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training, p=self.drop)
        x = self.fc2(x)
        x = F.log_softmax(x)
        x = x.view(x.size(0), 3, 2500)
        return x

    def forward_fc2(self, x):
        x = x.view(x.size(0), 1, 224, 224)
        x = self.conv1(x)
        x = F.max_pool2d(F.relu(self.bn1(x)), 2)
        x = self.rb1(x)
        x = self.rb2(x)
        x = self.rb3(x)
        x = self.rb4(x)
        x = self.rb5(x)
        x = self.rb6(x)
        x = self.rb7(x)
        x = self.rb8(x)
        x = self.maxpool(x)
        # x = self.deconv(x)
        # x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.size(0), self.n_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.view(x.size(0), 3, 2500)
        return x


class DECO_medium_conv(nn.Module):
    def __init__(self, drop=0.0, softmax=True):
        super(DECO_medium_conv, self).__init__()
        # TODO: pointNet.fc3 = torch.nn.Linear(pointNet.fc3.in_features, 51)
        self.classifier = pointnet.PointNetClassifier(k=16)
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
        self.n_size = self._get_conv_output((1, 224, 224))
        print("fc input size:" + str(self.n_size))
        # code.interact(local=locals())
        """
        self.fc1 = nn.Linear(self.n_size,4096)
        self.fc2 = nn.Linear(4096, 7500)
        """
        self.fc1 = nn.Linear(self.n_size, 4096)
        self.fc2 = nn.Linear(4096, 7500)
        # self.tanfc2 = nn.Hardtanh(min_val=-0.5, max_val=0.5, inplace=False, min_value=None, max_value=None)

    def _get_conv_output(self, shape):
        bs = 1
        input = Variable(torch.rand(bs, *shape))
        output_feat = self._forward_features(input)
        n_size = output_feat.data.view(bs, -1).size(1)
        print('OUTPUT_SHAPE = ' + str(output_feat.data.view(bs, -1).size()))
        return n_size

    def _forward_features(self, x):
        x = x.view(x.size(0), 1, 224, 224)
        x = self.conv1(x)
        x = F.max_pool2d(F.relu(self.bn1(x)), 2)
        x = self.rb1(x)
        x = self.rb2(x)
        x = self.maxpool1(x)
        x = self.resize_filters1(x)
        x = self.rb3(x)
        x = self.rb4(x)
        x = self.maxpool2(x)
        x = self.resize_filters2(x)
        x = self.rb5(x)
        x = self.rb6(x)
        x = self.maxpool3(x)
        x = self.resize_filters3(x)
        x = self.rb7(x)
        x = self.rb8(x)
        x = self.maxpool4(x)
        return x

    def forward_fc2(self, x):
        # print('input:' + str(x))
        x = x.view(x.size(0), 1, 224, 224)
        x = self.conv1(x)
        x = F.max_pool2d(F.relu(self.bn1(x)), 2)
        x = self.rb1(x)
        x = self.rb2(x)
        x = self.maxpool1(x)
        x = self.resize_filters1(x)
        x = self.rb3(x)
        x = self.rb4(x)
        x = self.maxpool2(x)
        x = self.resize_filters2(x)
        x = self.rb5(x)
        x = self.rb6(x)
        x = self.resize_filters3(x)
        x = self.maxpool3(x)
        x = self.rb7(x)
        x = self.rb8(x)
        x = self.maxpool4(x)
        x = x.view(x.size(0), self.n_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # x = self.tanfc2(x)
        # x = x-(x.max()+x.min())/2
        # x = x*0.5/x.max()
        # print('Xmin: '+ str(x.min())+' Xmax: '+ str(x.max()))
        x = x.view(x.size(0), 3, 2500)
        # print('After' + str(x))
        return x

    def forward(self, x):
        x = x.view(x.size(0), 1, 224, 224)
        x = self.conv1(x)
        x = F.max_pool2d(F.relu(self.bn1(x)), 2)
        x = self.rb1(x)
        x = self.rb2(x)
        x = self.maxpool1(x)
        x = self.resize_filters1(x)
        x = self.rb3(x)
        x = self.rb4(x)
        x = self.maxpool2(x)
        x = self.resize_filters2(x)
        x = self.rb5(x)
        x = self.rb6(x)
        x = self.resize_filters3(x)
        x = self.maxpool3(x)
        x = self.rb7(x)
        x = self.rb8(x)
        x = self.maxpool4(x)
        x = x.view(x.size(0), self.n_size)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training, p=self.drop)
        x = self.fc2(x)
        # x = self.tanfc2(x)
        # x = x-(x.max()+x.min())/2
        # x = x*0.5/x.max()
        # print('Xmin: '+ str(x.min())+' Xmax: '+ str(x.max()))
        if self.softmax:
            x = F.relu(x)
        x = x.view(x.size(0), 3, 2500)
        return self.classifier(x)


class DECO_senet(nn.Module):
    def __init__(self, drop=0.0, softmax=True):
        super(DECO_senet, self).__init__()
        # self.feat = PointNetCls(num_points = 2500, k = 51)
        self.feat = cls
        self.drop = drop
        self.softmax = softmax
        # self.feat = PointNetfeat(num_points = 2500, global_feat = True)
        self.in_planes = 16
        block = PreActBottleneck
        num_blocks = [1, 1, 1]
        filters = [16, 32, 64]
        self.conv1 = conv3x3(1, self.in_planes)
        self.stage1 = self._make_layer(block, filters[0], num_blocks[0], stride=1)
        self.stage2 = self._make_layer(block, filters[1], num_blocks[1], stride=2)
        self.stage3 = self._make_layer(block, filters[2], num_blocks[2], stride=2)
        self.maxpool1 = nn.AdaptiveAvgPool2d(1)
        self.n_size = self._get_conv_output((1, 224, 224))
        self.fc1 = nn.Linear(self.n_size, 4096)
        self.fc2 = nn.Linear(4096, 7500)
        print("block.expansion = " + str(block.expansion))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _get_conv_output(self, shape):
        bs = 1
        input = Variable(torch.rand(bs, *shape))
        output_feat = self._forward_features(input)
        n_size = output_feat.data.view(bs, -1).size(1)
        print('OUTPUT_SHAPE = ' + str(output_feat.data.view(bs, -1).size()))
        return n_size

    def _forward_features(self, x):
        x = x.view(x.size(0), 1, 224, 224)
        x = self.conv1(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.maxpool1(x)
        return x

    def forward(self, x):
        x = x.view(x.size(0), 1, 224, 224)
        x = self.conv1(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.maxpool1(x)
        x = x.view(x.size(0), self.n_size)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training, p=self.drop)
        x = self.fc2(x)
        if self.softmax:
            x = F.relu(x)
        x = x.view(x.size(0), 3, 2500)
        return self.feat(x)


class DECO_heavy_conv(nn.Module):
    def __init__(self, drop=0.0):
        super(DECO_heavy_conv, self).__init__()
        # self.feat = PointNetCls(num_points = 2500, k = 51)
        self.feat = cls
        self.drop = drop
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
        self.resize_filters3 = resize_filters(256, 7500)

        # self.rb7 = ResidualBlock(7500)
        # self.rb8 = ResidualBlock(512)
        # self.maxpool4 = nn.MaxPool2d(3,stride=2)
        self.n_size = self._get_conv_output((1, 224, 224))
        self.final_pool_size = np.sqrt(self.n_size / 7500).astype(int)
        print("final_pool_kernel_size: " + str(self.final_pool_size) + "x" + str(self.final_pool_size))
        self.maxpool4 = nn.MaxPool2d(self.final_pool_size, stride=1)
        # self.fc = nn.Linear(self.n_size, 7500)

    def _get_conv_output(self, shape):
        bs = 1
        input = Variable(torch.rand(bs, *shape))
        output_feat = self._forward_features(input)
        n_size = output_feat.data.view(bs, -1).size(1)
        shape = (output_feat.data.view(bs, -1).size(0), output_feat.data.view(bs, -1).size(1))
        print('OUTPUT_SHAPE = ' + str(shape[0]) + ', ' + str(shape[1]))
        return n_size

    def _forward_features(self, x):
        x = x.view(x.size(0), 1, 224, 224)
        x = self.conv1(x)
        x = F.max_pool2d(F.relu(self.bn1(x)), 2)
        x = self.rb1(x)
        x = self.rb2(x)
        x = self.maxpool1(x)
        x = self.resize_filters1(x)
        x = self.rb3(x)
        x = self.rb4(x)
        x = self.maxpool2(x)
        x = self.resize_filters2(x)
        x = self.rb5(x)
        x = self.rb6(x)
        x = self.maxpool3(x)
        x = self.resize_filters3(x)
        # x = self.rb7(x)
        # x = self.rb8(x)
        # x = self.maxpool4(x)
        return x

    def forward(self, x):
        x = x.view(x.size(0), 1, 224, 224)
        x = self.conv1(x)
        x = F.max_pool2d(F.relu(self.bn1(x)), 2)
        x = self.rb1(x)
        x = self.rb2(x)
        x = self.maxpool1(x)
        x = self.resize_filters1(x)
        x = self.rb3(x)
        x = self.rb4(x)
        x = self.maxpool2(x)
        x = self.resize_filters2(x)
        x = self.rb5(x)
        x = self.rb6(x)
        x = self.maxpool3(x)
        x = self.resize_filters3(x)
        # x = self.rb7(x)
        # x = self.rb8(x)
        x = self.maxpool4(x)
        # x = x.view(x.size(0), self.n_size)
        # x = F.relu(self.fc(x))
        x = F.dropout(x, training=self.training, p=self.drop)
        x = F.log_softmax(x)
        x = x.view(x.size(0), 3, 2500)
        return x

    def forward(self, x):
        x = x.view(x.size(0), 1, 224, 224)
        x = self.conv1(x)
        x = F.max_pool2d(F.relu(self.bn1(x)), 2)
        x = self.rb1(x)
        x = self.rb2(x)
        x = self.maxpool1(x)
        x = self.resize_filters1(x)
        x = self.rb3(x)
        x = self.rb4(x)
        x = self.maxpool2(x)
        x = self.resize_filters2(x)
        x = self.rb5(x)
        x = self.rb6(x)
        x = self.maxpool3(x)
        x = self.resize_filters3(x)
        # x = self.rb7(x)
        # x = self.rb8(x)
        x = self.maxpool4(x)
        # x = x.view(x.size(0), self.n_size)
        # x = F.relu(self.fc(x))
        x = F.dropout(x, training=self.training, p=self.drop)
        x = F.log_softmax(x)
        x = x.view(x.size(0), 3, 2500)
        return self.feat(x)


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

