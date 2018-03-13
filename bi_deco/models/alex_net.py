from __future__ import print_function

import torch.nn as nn
import torchvision.models


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, pretrained=True):
        super(AlexNet, self).__init__()
        self.alex_net = torchvision.models.alexnet(pretrained=pretrained)
        if pretrained:
            num_features = self.alex_net.classifier[6].in_features

            class_model = list(self.alex_net.classifier.children())
            class_model.pop()
            self.alex_net.classifier = nn.Sequential(*class_model)
            self.fc8 = nn.Linear(num_features, num_classes)

            for name, network_module in self.named_children():
                for param in network_module.parameters():
                    param.requires_grad = False
            self.fc8.requires_grad = True
        else:
            raise NotImplementedError()

    def forward(self, x):
        h = self.alex_net(x)
        y = self.fc8(h)
        return y

    def forward_fc7(self, x):
        raise NotImplementedError()
