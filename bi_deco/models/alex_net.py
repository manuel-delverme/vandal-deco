from __future__ import print_function

import torch.nn
import torchvision.models


class AlexNet(torch.nn.Module):
    def __init__(self, num_outputs=1000):
        super(AlexNet, self).__init__()
        self.alex_net = torchvision.models.alexnet(pretrained=True)
        num_features = self.alex_net.classifier[6].in_features

        class_model = list(self.alex_net.classifier.children())
        class_model.pop()
        self.alex_net.classifier = torch.nn.Sequential(*class_model)

        """
        self.fc71 = torch.nn.Linear(num_features, num_features, bias=True)
        torch.nn.init.xavier_uniform(self.fc71.weight)
        self.fc72 = torch.nn.Linear(num_features, num_features, bias=True)
        torch.nn.init.xavier_uniform(self.fc72.weight)
        self.fc73 = torch.nn.Linear(num_features, num_features, bias=True)
        torch.nn.init.xavier_uniform(self.fc73.weight)
        """

        self.fc8 = torch.nn.Linear(num_features, num_outputs, bias=True)
        torch.nn.init.xavier_uniform(self.fc8.weight)

        for name, network_module in self.named_children():
            for param in network_module.parameters():
                param.requires_grad = False
        self.fc8.requires_grad = True

    def forward(self, x):
        h = self.alex_net(x)
        y = self.fc8(h)
        return y

    def forward_fc7(self, x):
        h = self.alex_net(x)
        return h
