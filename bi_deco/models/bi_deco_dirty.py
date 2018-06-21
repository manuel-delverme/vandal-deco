from __future__ import print_function

import random
import torch
import torch.nn.parallel
import torch.nn
import torch.autograd
import torch.utils.data
import dirty_alexnet_wrapper as alex_net
import pointnet
import logging
import deco
import branch_dropout


class Bi_Deco(torch.nn.Module):
    def __init__(self, logger, dropout_probability=0.5, nr_points=2500, ensemble_hidden_size=2048, batch_norm2d=False,
                 bound_pointnet_deco=False, record_pcls=False, record_images=False, branch_dropout=False, from_scratch=False):
        WASHINGTON_CLASSES = 51
        super(Bi_Deco, self).__init__()
        self.logger = logger
        self.record_pcls = record_pcls
        self.record_images = record_images
        self.ensemble_hidden_size = ensemble_hidden_size
        self.branch_dropout = branch_dropout

        self.alexNet_classifier = alex_net.AlexNet()
        # num_feats = self.alexNet_classifier.model.Alex.classifier[6].in_features
        class_model = list(self.alexNet_classifier.model.Alex.classifier.children())
        class_model.pop()
        # class_model.append(nn.Linear(num_feats, num_classes))
        self.alexNet_classifier.model.Alex.classifier = torch.nn.Sequential(*class_model)

        # self.alexNet_deco = deco.DECO(is_alex_net=True, nr_points=nr_points, batch_norm2d=batch_norm2d)
        self.alexNet_deco = lambda x: x

        print("loading PointNet!")
        self.pointNet_classifier = pointnet.PointNetClassifier(
            num_points=nr_points,
            pretrained=not from_scratch,
            k=WASHINGTON_CLASSES
        )
        print("loading PointNet DECO")
        self.pointNet_deco = deco.DECO_pointNet_medium(
            nr_points=nr_points,
            bound_output=bound_pointnet_deco,
            # pretrained=not from_scratch
            # done below, manually for now..
        )
        if not from_scratch:
            pretrained_dict = torch.load("/home/alessandrodm/tesi/weights/split0/freezed/fc4096/epoch119DECO_medium_convNesterovfreezed.pkl")
            pointnet_clf = {}
            pointnet_deco = {}
            for k, v in pretrained_dict.items():
                if k.startswith("feat."):
                    pointnet_clf[k[5:]] = v
                else:
                    # if k.startswith("rb"):
                    #     print(k, v.shape)
                    #     k = "res{}".format(k[2:])
                    pointnet_deco[k] = v

            self.pointNet_classifier.load_state_dict(pointnet_clf)
            self.pointNet_deco.load_state_dict(pointnet_deco)

        self.dropout = torch.nn.Dropout(p=dropout_probability)

        alexnet_output_size = 4096
        if ensemble_hidden_size > 0:
            self.ensemble_fc = torch.nn.Linear(
                alexnet_output_size + self.pointNet_classifier.fc2.out_features,
                ensemble_hidden_size
            )
            self.ensemble_classifier = torch.nn.Linear(
                ensemble_hidden_size,
                WASHINGTON_CLASSES
            )
        else:
            self.ensemble_classifier_no_hidden = torch.nn.Linear(
                alexnet_output_size + self.pointNet_classifier.fc2.out_features,
                WASHINGTON_CLASSES
            )

    def forward(self, x):
        if self.branch_dropout and self.training:
            if random.random() > 0.5:
                use_pointnet, use_alexnet = True, False
            else:
                use_pointnet, use_alexnet = False, True
        else:
            use_pointnet, use_alexnet = True, True

        batch_size = x.shape[0]
        if use_alexnet:
            prediction_alexNet = self.alexNet_classifier.forward_fc7(x)
        else:
            prediction_alexNet = torch.autograd.Variable(
                torch.zeros((batch_size, self.alexNet_classifier.model.Alex.classifier[4].out_features))
            ).cuda()

        if use_pointnet:
            x_crop = x[:, :, 2:-2, 2:-2]
            point_cloud = self.pointNet_deco.forward_fc2(x_crop)
            prediction_pointNet = self.pointNet_classifier.forward_fc2(point_cloud)
        else:
            prediction_pointNet = torch.autograd.Variable(
                torch.zeros((batch_size, self.pointNet_classifier.fc2.out_features))
            ).cuda()

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
        else:
            # avoid dropping out neurons if we drop the whole branch
            # TODO: maybe dropout should still be applied?
            h_dropout = h_concat

        if self.ensemble_hidden_size > 0:
            h_ensemble = self.ensemble_fc(h_dropout)
            prediction = self.ensemble_classifier(h_ensemble)
        else:
            prediction = self.ensemble_classifier_no_hidden(h_dropout)

        return prediction
