from __future__ import print_function

import random
import torch
import torch.nn.parallel
import torch.utils.data
import alex_net
import pointnet
import logging
import deco


class Bi_Deco(torch.nn.Module):
    def __init__(self, logger, dropout_probability=0.5, nr_points=2500, ensemble_hidden_size=2048, batch_norm2d=False,
                 bound_pointnet_deco=False, record_pcls=False, record_images=False, branch_dropout=False):
        WASHINGTON_CLASSES = 51
        super(Bi_Deco, self).__init__()
        self.logger = logger

        self.record_pcls = record_pcls
        self.record_images = record_images
        self.ensemble_hidden_size = ensemble_hidden_size
        self.branch_dropout = branch_dropout

        print("loading AlexNet")
        self.alexNet_classifier = alex_net.AlexNet(num_outputs=4069, pretrained=True, )
        print("loading AlexNet deco")
        self.alexNet_deco = deco.DECO(is_alex_net=True, nr_points=nr_points, batch_norm2d=batch_norm2d)

        print("loading PointNet")
        self.pointNet_classifier = pointnet.PointNetClassifier(num_points=nr_points, pretrained=True, k=WASHINGTON_CLASSES)
        print("loading PointNet DECO")
        self.pointNet_deco = deco.DECO(
            is_alex_net=False, nr_points=nr_points, batch_norm2d=batch_norm2d, bound_output=bound_pointnet_deco)

        self.dropout = torch.nn.Dropout(p=dropout_probability)

        if ensemble_hidden_size > 0:
            self.ensemble_fc = torch.nn.Linear(
                4096 + self.pointNet_classifier.fc2.out_features,
                ensemble_hidden_size
            )
            self.ensemble_classifier = torch.nn.Linear(
                ensemble_hidden_size,
                WASHINGTON_CLASSES
            )
        else:
            self.ensemble_classifier_no_hidden = torch.nn.Linear(
                4096 + self.pointNet_classifier.fc2.out_features,
                WASHINGTON_CLASSES
            )

    def forward(self, x, train=True):
        if self.branch_dropout and train:
            if random.random() > 0.5:
                use_pointnet, use_alexnet = True, False
            else:
                use_pointnet, use_alexnet = False, True
        else:
            use_pointnet, use_alexnet = True, True

        if use_alexnet:
            augmented_image = self.alexNet_deco(x)
            prediction_alexNet = self.alexNet_classifier.forward_fc7(augmented_image)
        else:
            prediction_alexNet = torch.zeros()# ???) #  TODO: fill(0)

        if use_pointnet:
            point_cloud = self.pointNet_deco(x)
            prediction_pointNet = self.pointNet_classifier.forward_fc2(point_cloud)
        else:
            prediction_pointNet = torch.zeros()# ???) #  TODO: fill(0)

        if self.record_pcls:
            raise NotImplementedError()
            cloud = point_cloud.tolist()
            with open(self.record_pcls, "ab") as fout:
                fout.write(",".join(cloud))

        if self.record_images:
            raise NotImplementedError()
            self.logger.image_summary("deco_image", prediction_alexNet.tolist(), -1)

        h_concat = torch.cat((prediction_alexNet, prediction_pointNet), dim=1)

        if not self.branch_dropout:
            h_dropout = self.dropout(h_concat)

        if self.ensemble_hidden_size > 0:
            h_ensemble = self.ensemble_fc(h_dropout)
            prediction = self.ensemble_classifier(h_ensemble)
        else:
            prediction = self.ensemble_classifier_no_hidden(h_dropout)

        return prediction


