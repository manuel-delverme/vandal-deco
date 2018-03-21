from __future__ import print_function

import torch
import torch.nn.parallel
import torch.utils.data
import alex_net
import pointnet
import deco


class Bi_Deco(torch.nn.Module):
    def __init__(self, dropout_probability=0.5, nr_points=2500, ensemble_hidden_size=2048):
        WASHINGTON_CLASSES = 51
        super(Bi_Deco, self).__init__()

        self.alexNet_classifier = alex_net.AlexNet(num_classes=WASHINGTON_CLASSES, pretrained=True, )
        self.alexNet_deco = deco.DECO(is_alex_net=True, nr_points=nr_points)

        self.pointNet_classifier = pointnet.PointNetClassifier(num_points=nr_points, pretrained=True,
                                                                              k=WASHINGTON_CLASSES)
        self.pointNet_deco = deco.DECO(is_alex_net=False, nr_points=nr_points)

        self.dropout = torch.nn.Dropout(p=dropout_probability)

        self.ensemble_fc = torch.nn.Linear(
            self.alexNet_classifier.fc8.out_features + self.pointNet_classifier.fc3.out_features,
            ensemble_hidden_size
        )
        self.ensemble_classifier = torch.nn.Linear(
            ensemble_hidden_size,
            WASHINGTON_CLASSES
        )
        self.ensemble_classifier_no_hidden = torch.nn.Linear(
            self.alexNet_classifier.fc8.out_features + self.pointNet_classifier.fc3.out_features,
            WASHINGTON_CLASSES
        )

    def forward(self, x):
        augmented_image = self.alexNet_deco(x)
        prediction_alexNet = self.alexNet_classifier(augmented_image)

        point_cloud = self.pointNet_deco(x)
        prediction_pointNet = self.pointNet_classifier(point_cloud)

        h_concat = torch.cat((prediction_alexNet, prediction_pointNet), dim=1)

        h_dropout = self.dropout(h_concat)
        h_ensemble = self.ensemble_fc(h_dropout)
        prediction = self.ensemble_classifier(h_ensemble)
        return prediction



