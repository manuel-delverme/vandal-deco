from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
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


class DECO_alexNet(nn.Module):
    def __init__(self, bound_output=False):
        super(DECO_alexNet, self).__init__()
        self.bound_output = bound_output
        self.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)

        self.Lrelu = nn.LeakyReLU(negative_slope=0.01)
        self.pool = nn.MaxPool2d(3, stride=2)  # 64x57x57
        self.res1 = ResidualBlock(64)
        self.res2 = ResidualBlock(64)
        self.res3 = ResidualBlock(64)
        self.res4 = ResidualBlock(64)
        self.res5 = ResidualBlock(64)
        self.res6 = ResidualBlock(64)
        self.res7 = ResidualBlock(64)
        self.res8 = ResidualBlock(64)

        self.conv2 = nn.Conv2d(64, 3, 1, stride=1)
        self.deconv_to_image = nn.ConvTranspose2d(3, 3, 8, stride=4, padding=0, groups=3, bias=False)
        if bound_output:
            self.output_bound = nn.Sigmoid()

    def forward(self, x):
        batch_size = x.size(0)
        # TODO: is this useless?
        x = x.view(batch_size, 1, 228, 228)
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.Lrelu(h)
        h = self.pool(h)
        h = self.res1(h)
        h = self.res2(h)
        h = self.res3(h)
        h = self.res4(h)
        h = self.res5(h)
        h = self.res6(h)
        h = self.res7(h)
        h = self.res8(h)

        h = self.conv2(h)
        h = self.deconv_to_image(h)

        if self.bound_output:
            h = self.output_bound(h)
        return h


class DECO_pointNet(nn.Module):
    def __init__(self, nr_points=None, pretrained=False, bound_output=False):
        super(DECO_pointNet, self).__init__()
        self.nr_points = nr_points
        # the idea was dropped
        self.bound_output = bound_output

        self.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3)

        self.bn1 = nn.BatchNorm2d(64)
        self.Lrelu = nn.LeakyReLU(negative_slope=0.01)
        self.pool = nn.MaxPool2d(3, stride=2)  # 64x57x57
        self.res1 = ResidualBlock(64)
        self.res2 = ResidualBlock(64)
        self.res3 = ResidualBlock(64)
        self.res4 = ResidualBlock(64)
        self.res5 = ResidualBlock(64)
        self.res6 = ResidualBlock(64)
        self.res7 = ResidualBlock(64)
        self.res8 = ResidualBlock(64)

        self.bn2 = nn.BatchNorm2d(64)
        self.last_pool = nn.MaxPool2d(3, stride=2)

        self.parameter_killer1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.parameter_killer2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.parameter_killer3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # self.fc1 = nn.Linear(64 * 27 * 27, 4096)
        self.fc_to_3d_points = nn.Linear(512 * 6 * 6, self.nr_points * 3)
        if pretrained:
            pretrained_dict = torch.load(pretrained)
            self.load_state_dict(pretrained_dict)
            for name, network_module in self.named_children():
                for param in network_module.parameters():
                    param.requires_grad = False
        if bound_output:
            self.output_bound = nn.Tanh()

    def forward(self, x):
        batch_size = x.size(0)
        # TODO: is this useless?
        x = x.view(batch_size, 1, 224, 224)
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.Lrelu(h)
        h = self.pool(h)
        h = self.res1(h)
        h = self.res2(h)
        h = self.res3(h)
        h = self.res4(h)
        h = self.res5(h)
        h = self.res6(h)
        h = self.res7(h)
        h = self.res8(h)

        h = self.parameter_killer1(h)
        h = self.parameter_killer2(h)
        h = self.parameter_killer3(h)

        # h = self.last_pool(h)
        h = h.view(batch_size, -1)

        # h = F.relu(self.fc1(h))
        h = self.fc_to_3d_points(h)
        # h = F.log_softmax(h)
        h = h.view(batch_size, 3, self.nr_points)

        if self.bound_output:
            h = self.output_bound(h)
        return h
