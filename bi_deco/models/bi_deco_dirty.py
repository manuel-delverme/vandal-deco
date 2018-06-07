from __future__ import print_function

import random
import torch
import torch.nn.parallel
import torch.utils.data
import dirty_alexnet_wrapper as alex_net
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

        print("loading AlexNet + deco")
        self.alexNet_classifier = alex_net.AlexNet(num_outputs=4069, pretrained=True, )
        del self.alexNet_classifier.model.Alex.classifier
        # print("loading AlexNet deco")

        # self.alexNet_deco = deco.DECO(is_alex_net=True, nr_points=nr_points, batch_norm2d=batch_norm2d)
        self.alexNet_deco = lambda x: x

        print("loading PointNet")
        self.pointNet_classifier = pointnet.PointNetClassifier(num_points=nr_points, pretrained=True,
                                                               k=WASHINGTON_CLASSES)
        print("loading PointNet DECO")
        self.pointNet_deco = deco.DECO_pointNet(nr_points=nr_points, bound_output=bound_pointnet_deco)

        self.dropout = torch.nn.Dropout(p=dropout_probability)

        if ensemble_hidden_size > 0:
            self.ensemble_fc = torch.nn.Linear(
                6400 + self.pointNet_classifier.fc2.out_features,
                ensemble_hidden_size
            )
            self.ensemble_classifier = torch.nn.Linear(
                ensemble_hidden_size,
                WASHINGTON_CLASSES
            )
        else:
            self.ensemble_classifier_no_hidden = torch.nn.Linear(
                6400 + self.pointNet_classifier.fc2.out_features,
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

        """
        @staticmethod
        def _make_noise(input):
            return input.new().resize_as_(input)

        @classmethod
        def forward(cls, ctx, input, p=0.5, train=False, inplace=False):
            if p < 0 or p > 1:
                raise ValueError("dropout probability has to be between 0 and 1, "
                                 "but got {}".format(p))
            ctx.p = p
            ctx.train = train
            ctx.inplace = inplace

            if ctx.inplace:
                ctx.mark_dirty(input)
                output = input
            else:
                output = input.clone()

        if ctx.p > 0 and ctx.train:
            ctx.noise = cls._make_noise(input)
            if ctx.p == 1:
                ctx.noise.fill_(0)
            else:
                ctx.noise.bernoulli_(1 - ctx.p).div_(1 - ctx.p)
            ctx.noise = ctx.noise.expand_as(input)
            output.mul_(ctx.noise)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.p > 0 and ctx.train:
            return grad_output.mul(Variable(ctx.noise)), None, None, None
        else:
            return grad_output, None, None, None
        """

        if use_alexnet:
            augmented_image = self.alexNet_deco(x)
            prediction_alexNet = self.alexNet_classifier.forward_fc7(augmented_image)
        else:
            # input.new().resize_as_(input)
            # ctx.noise = cls._make_noise(input)
            # ctx.noise.fill_(0)
            prediction_alexNet = torch.zeros()  # ???) #  TODO: fill(0)

        if use_pointnet:
            point_cloud = self.pointNet_deco(x)
            prediction_pointNet = self.pointNet_classifier.forward_fc2(point_cloud)
        else:
            prediction_pointNet = torch.zeros()  # ???) #  TODO: fill(0)

        if self.record_pcls:
            raise NotImplementedError()
            cloud = point_cloud.tolist()
            with open(self.record_pcls, "ab") as fout:
                fout.write(",".join(cloud))

        if self.record_images:
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
