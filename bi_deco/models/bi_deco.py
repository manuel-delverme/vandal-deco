from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data

import deco
import pointnet
import alex_net


class BiDeco(nn.Module):
    def __init__(self, dropout_probability=0.5):
        WASHINGTON_CLASSES = 51

        super(BiDeco, self).__init__()
        self.alexNet_deco = deco.DECO(alex_net=True)
        # self.alexNet_classifier = alex_net.AlexNet()
        self.alexNet_classifier = alex_net.alexnet(pretrained=True)

        self.pointNet_deco = deco.DECO(alex_net=False)
        self.pointNet_classifier = pointnet.PointNetClassifier(k=WASHINGTON_CLASSES, pretrained=True)

        self.dropout = torch.nn.Dropout(p=dropout_probability)

        # as we discussed (use the last layer)
        # not using 5th layer MISLEADING, forward uses RELU
        self.ensemble = torch.nn.Linear(
            self.alexNet_classifier.classifier[4].out_features + self.pointNet_classifier.fc2.out_features,
            WASHINGTON_CLASSES
        )

        train_nets = [
            self.alexNet_deco,
            self.pointNet_deco,
            self.ensemble,
        ]
        frozen_net = [
            self.alexNet_classifier,
            self.pointNet_classifier,
        ]
        for net in train_nets:
            for name, network_module in net.named_children():
                for param in network_module.parameters():
                    param.requires_grad = True

        for net in frozen_net:
            for name, network_module in net.named_children():
                for param in network_module.parameters():
                    param.requires_grad = False

        # self.alexNet_classifier.classifier[4].requires_grad = True
        # self.pointNet_classifier.fc2.requires_grad = True

        # self.dropout = F.dropout()

    def forward(self, x):
        # /home/deco2/python/Alex.py
        h_alex = self.alexNet_deco(x)
        # print(h_alex.size())
        h_alex = self.alexNet_classifier(h_alex)
        # print(h_alex.size())
        # print("POINTNET\t" * 10)

        h_pointnet = self.pointNet_deco(x)
        batch_size, dim1 = h_pointnet.size()
        h_pointnet = h_pointnet.view(batch_size, 3, 2500)
        h_pointnet, some_trash = self.pointNet_classifier(h_pointnet)
        # TODO: double check for activation in both pointnet and alexNet

        h_concat = torch.cat((h_alex, h_pointnet), dim=1)

        # TODO: remove? TRY IT OUT
        # h_dropout = self.dropout(h_concat)
        h_dropout = h_concat

        prediction = self.ensemble(h_dropout)

        # y_hat = F.log_softmax(prediction)
        # return y_hat
        return prediction

        # x = x.view(x.size(0), 1, 224, 224)
        # x = x.view(x.size(0), self.n_size)
        # x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training, p=self.drop)
