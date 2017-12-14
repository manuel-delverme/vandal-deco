from __future__ import print_function
import deco
import pointnet
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch
import torch.nn.functional as F


class BiDeco(nn.Module):
    def __init__(self, dropout_probability=0.0):
        super(BiDeco, self).__init__()
        self.alexNet_deco = deco.DECO_medium_conv()
        self.alexNet_classifier = alexNet.AlexNet()
        # TODO: freeze stuff in alexNet_classifier
        self.pointNet_deco = deco.DECO_medium_conv()
        self.pointNet_classifier = pointnet.PointNetClassifier(k=16)
        # TODO: freeze stuff in pointnetClassifier
        # TODO: pop pointNet.fc3
        # TODO: fix forward
        # self.dropout = F.dropout()
        # TODO: try adding more layers
        self.dropout = torch.nn.Dropout(p=dropout_probability)
        self.pre_ensemble = torch.nn.Linear(self.alexNet_classifier.fc__.out + self.pointNet_classifier.fc__.out, 256)
        # self.ensemble = torch.nn.Linear(self.pre_ensemble., num_classes)

    def forward(self, x):
        # /home/deco2/python/Alex.py
        h_alex = self.alexNet_deco(x)
        h_alex = self.alexNet_classifier(h_alex)

        h_pointnet = self.pointNet_deco(x)
        h_pointnet = self.pointNet_classifier(h_pointnet)
        # TODO: double check for activation in both pointnet and alexNet

        h_concat = torch.cat((h_alex, h_pointnet), dim=1)
        h_dropout = self.dropout(h_concat)
        prediction = self.ensemble(h_dropout)

        y_hat = F.log_softmax(prediction)
        return y_hat
        # x = x.view(x.size(0), 1, 224, 224)
        # x = x.view(x.size(0), self.n_size)
        # x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training, p=self.drop)
