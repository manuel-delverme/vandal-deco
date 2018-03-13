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
    def __init__(self, dropout_probability, nr_points):
        self.nr_poitnet_points = nr_points
        WASHINGTON_CLASSES = 51

        super(BiDeco, self).__init__()
        self.alexNet_deco = bi_deco.models.deco.DECO(alex_net=True, nr_points=nr_points)
        self.alexNet_classifier = alex_net.AlexNet(pretrained=True)

        self.pointNet_deco = bi_deco.models.deco.DECO(alex_net=False, nr_points=nr_points)
        self.pointNet_classifier = pointnet.PointNetClassifier(k=1000, pretrained=True)

        self.dropout = torch.nn.Dropout(p=dropout_probability)

        self.ensemble = torch.nn.Linear(
            self.alexNet_classifier.classifier[6].out_features + self.pointNet_classifier.fc3.out_features,
            WASHINGTON_CLASSES
        )

    def forward(self, x):
        h_alex = self.alexNet_deco(x)
        h_alex = self.alexNet_classifier(h_alex)

        point_cloud = self.pointNet_deco(x)
        h_pointnet = self.pointNet_classifier(point_cloud)

        h_concat = torch.cat((h_alex, h_pointnet), dim=1)

        h_dropout = self.dropout(h_concat)
        prediction = self.ensemble(h_dropout)
        return prediction
