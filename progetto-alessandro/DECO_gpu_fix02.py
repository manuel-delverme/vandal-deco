
from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pdb
import torch.nn.functional as F
from torch.nn.modules.module import _addindent
import code
from senet import SeResNet, PreActBottleneck, PreActBlock, conv3x3

def get_output_shape(model,module):
    if not module._modules.keys:
        last_layer_name=module._modules.keys()[-1]
        last_layer = module._modules.get(last_layer_name)
        h = last_layer.register_forward_hook(
        lambda m, i, o: \
        print(
        'm:', type(m),
        '\ni:', type(i),
        '\n   len:', len(i),
        '\n   type:', type(i[0]),
       '\n   data size:', i[0].data.size(),
       '\n   data type:', i[0].data.type(),
        '\no:', type(o),
       '\n   data size:', o.data.size(),
       '\n   data type:', o.data.type(),
            )
            )
        #h.remove()
        x = Variable(torch.randn(1, 1, 224, 224)).cuda()
        h_x = model(x)
#    code.interact(local=locals())

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
        total_params+=params
        tmpstr += '  (' + key + '): ' + modstr 
        if show_weights:
            tmpstr += ', weights={}'.format(weights)
        if show_parameters:
            tmpstr +=  ', parameters={}'.format(params)
        tmpstr += '\n'
        #get_output_shape(model,module)
        #if hasattr(module,'output'):   
        #    tmpstr += ' output_shape={}'.format(module.output.size())
        #    tmpstr += '\n'   
    tmpstr = tmpstr + ')'
    tmpstr = tmpstr + '\n total number of parameters={}'.format(total_params)
    return tmpstr

class STN3d(nn.Module):
    def __init__(self, num_points = 2500):
        super(STN3d, self).__init__()
        self.num_points = num_points
        
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.mp1 = torch.nn.MaxPool1d(num_points)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.mp1(x)
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x



class PointNetfeat(nn.Module):
    def __init__(self, num_points = 2500, global_feat = True):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d(num_points = num_points)
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.mp1 = torch.nn.MaxPool1d(num_points)
        self.num_points = num_points
        self.global_feat = global_feat



    def forward(self, x):
        batchsize = x.size()[0]
        trans = self.stn(x)
        x = x.transpose(2,1)
        x = torch.bmm(x, trans)
        x = x.transpose(2,1)
        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = self.mp1(x)
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
            return torch.cat([x, pointfeat], 1), trans



class PointNetCls(nn.Module):
    def __init__(self, num_points = 2500, k = 16):
        super(PointNetCls, self).__init__()
        self.num_points = num_points
        self.feat = PointNetfeat(num_points, global_feat=True)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        #num_ftrs += cls.fc3.in_features
        #cls.fc3 = nn.Linear(num_ftrs,51)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
    def forward(self, x):
        x, trans = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return F.log_softmax(x), trans 


cls = PointNetCls(k = 16)
#cls.load_state_dict(torch.load('./pointnet_weights/cls/cls_model_24.pth'))
cls.load_state_dict(torch.load('./pointnet_weights/cls/cls_model_24.pth', map_location=lambda storage, loc: storage))

num_ftrs = cls.fc3.in_features
print('num:'+str(num_ftrs))
print(cls.fc3)
cls.fc3 = nn.Linear(num_ftrs, 51)
print(cls.fc3)

class ResidualBlock(nn.Module):
    def __init__(self,filters,pad=1):
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
    def __init__(self,filters_in,filters_out):
        super(resize_filters, self).__init__()
        self.conv1 = nn.Conv2d(filters_in, filters_out, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(filters_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        return out


class DECO(nn.Module):
    def __init__(self,drop=0.0):
        super(DECO, self).__init__()
        #self.feat = PointNetCls(num_points = 2500, k = 51)
        self.feat = cls
        self.drop = drop
        #self.feat = PointNetfeat(num_points = 2500, global_feat = True)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7,stride = 2,padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.rb1 = ResidualBlock(64)
        self.rb2 = ResidualBlock(64)
        self.rb3 = ResidualBlock(64)
        self.rb4 = ResidualBlock(64)
        self.rb5 = ResidualBlock(64)
        self.rb6 = ResidualBlock(64)
        self.rb7 = ResidualBlock(64)
        self.rb8 = ResidualBlock(64)
        self.maxpool = nn.MaxPool2d(3,stride=2)
        self.n_size = self._get_conv_output((1,224,224))
        print ("fc input size:"+str(self.n_size))
        #code.interact(local=locals())

        self.fc1 = nn.Linear(self.n_size,4096)
        self.fc2 = nn.Linear(4096, 7500)

    def _get_conv_output(self, shape):
        bs = 1
        input = Variable(torch.rand(bs, *shape))
        output_feat = self._forward_features(input)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def _forward_features(self, x):
        x = x.view(x.size(0), 1,224,224)
        x = self.conv1(x)
        x = F.max_pool2d(F.relu(self.bn1(x)),2)
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
        x = x.view(x.size(0), 1,224,224)
        x = self.conv1(x)
        x = F.max_pool2d(F.relu(self.bn1(x)),2)
        x = self.rb1(x)
        x = self.rb2(x)
        x = self.rb3(x)
        x = self.rb4(x)
        x = self.rb5(x)
        x = self.rb6(x)
        x = self.rb7(x)
        x = self.rb8(x)
        x = self.maxpool(x)
        #x = self.deconv(x)
        #x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.size(0), self.n_size)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training,p=self.drop)
        x = self.fc2(x)
        x = F.log_softmax(x)
        x = x.view(x.size(0),3, 2500)
        return self.feat(x)

    def forward_fc2(self, x):
        x = x.view(x.size(0), 1,224,224)
        x = self.conv1(x)
        x = F.max_pool2d(F.relu(self.bn1(x)),2)
        x = self.rb1(x)
        x = self.rb2(x)
        x = self.rb3(x)
        x = self.rb4(x)
        x = self.rb5(x)
        x = self.rb6(x)
        x = self.rb7(x)
        x = self.rb8(x)
        x = self.maxpool(x)
        #x = self.deconv(x)
        #x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.size(0), self.n_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.view(x.size(0),3, 2500)
        return x



class DECO_medium_conv(nn.Module):
    def __init__(self,drop=0.0,softmax=True):
        super(DECO_medium_conv, self).__init__()
        #self.feat = PointNetCls(num_points = 2500, k = 51)
        self.feat = cls
        self.drop = drop
        self.softmax = softmax
        #self.feat = PointNetfeat(num_points = 2500, global_feat = True)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7,stride = 2,padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.rb1 = ResidualBlock(64)
        self.rb2 = ResidualBlock(64)
        self.maxpool1 = nn.MaxPool2d(3,stride=2)
        self.resize_filters1 = resize_filters(64,128)
        self.rb3 = ResidualBlock(128)
        self.rb4 = ResidualBlock(128)
        self.maxpool2 = nn.MaxPool2d(3,stride=2)
        self.resize_filters2 = resize_filters(128,256)
        self.rb5 = ResidualBlock(256)
        self.rb6 = ResidualBlock(256)
        self.maxpool3 = nn.MaxPool2d(3,stride=2)
        self.resize_filters3 = resize_filters(256,512)
        self.rb7 = ResidualBlock(512)
        self.rb8 = ResidualBlock(512)
        self.maxpool4 = nn.MaxPool2d(3,stride=2)
        self.n_size = self._get_conv_output((1,224,224))
        print ("fc input size:"+str(self.n_size))
        #code.interact(local=locals())
        """
        self.fc1 = nn.Linear(self.n_size,4096)
        self.fc2 = nn.Linear(4096, 7500)
        """
        self.fc1 = nn.Linear(self.n_size,4096)
        self.fc2 = nn.Linear(4096, 7500)
        #self.tanfc2 = nn.Hardtanh(min_val=-0.5, max_val=0.5, inplace=False, min_value=None, max_value=None)
    def _get_conv_output(self, shape):
        bs = 1
        input = Variable(torch.rand(bs, *shape))
        output_feat = self._forward_features(input)
        n_size = output_feat.data.view(bs, -1).size(1)
        print ('OUTPUT_SHAPE = '+str(output_feat.data.view(bs, -1).size()))
        return n_size

    def _forward_features(self, x):
        x = x.view(x.size(0), 1,224,224)
        x = self.conv1(x)
        x = F.max_pool2d(F.relu(self.bn1(x)),2)
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
        #print('input:' + str(x))
        x = x.view(x.size(0), 1,224,224)
        x = self.conv1(x)
        x = F.max_pool2d(F.relu(self.bn1(x)),2)
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
        #x = self.tanfc2(x)
        #x = x-(x.max()+x.min())/2
        #x = x*0.5/x.max()
        #print('Xmin: '+ str(x.min())+' Xmax: '+ str(x.max()))
        x = x.view(x.size(0), 3, 2500)
        #print('After' + str(x))
        return x


    def forward(self, x):
        x = x.view(x.size(0), 1,224,224)
        x = self.conv1(x)
        x = F.max_pool2d(F.relu(self.bn1(x)),2)
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
        x = F.dropout(x, training=self.training,p=self.drop)
        x = self.fc2(x)
        #x = self.tanfc2(x)
        #x = x-(x.max()+x.min())/2
        #x = x*0.5/x.max()
        #print('Xmin: '+ str(x.min())+' Xmax: '+ str(x.max()))
        if self.softmax:
            x = F.relu(x)
        x = x.view(x.size(0),3, 2500)
        return self.feat(x)

class DECO_senet(nn.Module):
    def __init__(self,drop=0.0,softmax=True):
        super(DECO_senet, self).__init__()
        #self.feat = PointNetCls(num_points = 2500, k = 51)
        self.feat = cls
        self.drop = drop
        self.softmax = softmax
        #self.feat = PointNetfeat(num_points = 2500, global_feat = True)
        self.in_planes = 16
        block = PreActBottleneck
        num_blocks = [1, 1, 1]
        filters = [16, 32, 64]
        self.conv1 = conv3x3(1, self.in_planes)
        self.stage1 = self._make_layer(block, filters[0], num_blocks[0], stride=1)
        self.stage2 = self._make_layer(block, filters[1], num_blocks[1], stride=2)
        self.stage3 = self._make_layer(block, filters[2], num_blocks[2], stride=2)
        self.maxpool1 = nn.AdaptiveAvgPool2d(1)
        self.n_size = self._get_conv_output((1,224,224))
        self.fc1 = nn.Linear(self.n_size,4096)
        self.fc2 = nn.Linear(4096, 7500)
        print ("block.expansion = " + str(block.expansion))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
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
        print ('OUTPUT_SHAPE = '+str(output_feat.data.view(bs, -1).size()))
        return n_size

    def _forward_features(self, x):
        x = x.view(x.size(0), 1,224,224)
        x = self.conv1(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.maxpool1(x)
        return x

    def forward(self, x):
        x = x.view(x.size(0), 1,224,224)
        x = self.conv1(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.maxpool1(x)
        x = x.view(x.size(0), self.n_size)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training,p=self.drop)
        x = self.fc2(x)
        if self.softmax:
            x = F.relu(x)
        x = x.view(x.size(0),3, 2500)
        return self.feat(x)

class DECO_heavy_conv(nn.Module):
    def __init__(self,drop=0.0):
        super(DECO_heavy_conv, self).__init__()
        #self.feat = PointNetCls(num_points = 2500, k = 51)
        self.feat = cls
        self.drop = drop
        #self.feat = PointNetfeat(num_points = 2500, global_feat = True)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7,stride = 2,padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.rb1 = ResidualBlock(64)
        self.rb2 = ResidualBlock(64)
        self.maxpool1 = nn.MaxPool2d(3,stride=2)
        self.resize_filters1 = resize_filters(64,128)
        self.rb3 = ResidualBlock(128)
        self.rb4 = ResidualBlock(128)
        self.maxpool2 = nn.MaxPool2d(3,stride=2)
        self.resize_filters2 = resize_filters(128,256)
        self.rb5 = ResidualBlock(256)
        self.rb6 = ResidualBlock(256)
        self.maxpool3 = nn.MaxPool2d(3,stride=2)
        self.resize_filters3 = resize_filters(256,7500)

        #self.rb7 = ResidualBlock(7500)
        #self.rb8 = ResidualBlock(512)
        #self.maxpool4 = nn.MaxPool2d(3,stride=2)
        self.n_size = self._get_conv_output((1,224,224))
        self.final_pool_size = np.sqrt(self.n_size / 7500).astype(int)
        print ("final_pool_kernel_size: "+str(self.final_pool_size)+"x"+str(self.final_pool_size))
        self.maxpool4 = nn.MaxPool2d(self.final_pool_size,stride=1)
        #self.fc = nn.Linear(self.n_size, 7500)
    def _get_conv_output(self, shape):
        bs = 1
        input = Variable(torch.rand(bs, *shape))
        output_feat = self._forward_features(input)
        n_size = output_feat.data.view(bs, -1).size(1)
        shape = (output_feat.data.view(bs, -1).size(0),output_feat.data.view(bs, -1).size(1))
        print ('OUTPUT_SHAPE = '+str(shape[0])+', '+str(shape[1]))
        return n_size

    def _forward_features(self, x):
        x = x.view(x.size(0), 1,224,224)
        x = self.conv1(x)
        x = F.max_pool2d(F.relu(self.bn1(x)),2)
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
        #x = self.rb7(x)
        #x = self.rb8(x)
        #x = self.maxpool4(x)
        return x

    def forward(self, x):
        x = x.view(x.size(0), 1,224,224)
        x = self.conv1(x)
        x = F.max_pool2d(F.relu(self.bn1(x)),2)
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
        #x = self.rb7(x)
        #x = self.rb8(x)
        x = self.maxpool4(x)
        #x = x.view(x.size(0), self.n_size)
        #x = F.relu(self.fc(x))
        x = F.dropout(x, training=self.training,p=self.drop)
        x = F.log_softmax(x)
        x = x.view(x.size(0),3, 2500)
        return x


    def forward(self, x):
        x = x.view(x.size(0), 1,224,224)
        x = self.conv1(x)
        x = F.max_pool2d(F.relu(self.bn1(x)),2)
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
        #x = self.rb7(x)
        #x = self.rb8(x)
        x = self.maxpool4(x)
        #x = x.view(x.size(0), self.n_size)
        #x = F.relu(self.fc(x))
        x = F.dropout(x, training=self.training,p=self.drop)
        x = F.log_softmax(x)
        x = x.view(x.size(0),3, 2500)
        return self.feat(x)
   

class PointNetDenseCls(nn.Module):
    def __init__(self, num_points = 2500, k = 51):
        super(PointNetDenseCls, self).__init__()
        self.num_points = num_points
        self.k = k
        self.feat = PointNetfeat(num_points, global_feat=False)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]
        x, trans = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2,1).contiguous()
        x = F.log_softmax(x.view(-1,self.k))
        x = x.view(batchsize, self.num_points, self.k)
        return x, trans

"""
if __name__ == '__main__':
    sim_data = Variable(torch.rand(32,3,2500))
    trans = STN3d()
    out = trans(sim_data)
    print('stn', out.size())

    pointfeat = PointNetfeat(global_feat=True)
    out, _ = pointfeat(sim_data)
    print('global feat', out.size())

    pointfeat = PointNetfeat(global_feat=False)
    out, _ = pointfeat(sim_data)
    print('point feat', out.size())

    cls = PointNetCls(k = 5)
    
    out, _ = cls(sim_data)
    print('class', out.size())

    seg = PointNetDenseCls(k = 3)
    out, _ = seg(sim_data)
    print('seg', out.size())


"""
