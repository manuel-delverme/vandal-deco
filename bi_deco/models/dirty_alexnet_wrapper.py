import classDecoAlex_onlyDepth
import torch.nn


class AlexNet(torch.nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()

        model_path = '/home/iodice/vandal-deco/working_alexnet_weights_best_735.pkl'
        model = classDecoAlex_onlyDepth.DecoAlexNet(num_classes=51).cuda()


        # for layer in model.Deco.named_children():
        #     for param in layer[1].parameters():
        #         param.requires_grad = True

        model.load_state_dict(torch.load(model_path))
        for param in model.parameters():
            param.requires_grad = False

        self.model = model

    def forward(self, x):
        h = self.model(x)
        return h

    def forward_fc7(self, x):
        return self.model.forward_fc7(x)
