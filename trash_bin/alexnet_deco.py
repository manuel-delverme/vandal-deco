from __future__ import print_function
from __future__ import print_function

import os
import tqdm

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# is this even working maybe it has to be declared earlier
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # CPU

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
import bi_deco.models.alex_net
import pickle
import collections
import numpy as np
import matplotlib.pyplot as plt
import bi_deco.models.pointnet
import bi_deco.models.bi_deco
import bi_deco.datasets.washington
import torch
import torch.nn.parallel
import torch.utils.data
import argparse
import os
import torch.optim as optim
import torch.utils.data
import torch.nn
from torch.autograd import Variable

RESOURCES_HOME = "/home/alessandrodm/tesi/"
RESULTS_HOME = "/home/iodice/alessandro_results/"


class Alexnet_only(nn.Module):
    def __init__(self, dropout_probability=0.5):
        WASHINGTON_CLASSES = 51
        super(Alexnet_only, self).__init__()

        self.alexNet_classifier = bi_deco.models.alex_net.alexnet(pretrained=True)
        self.ensemble = torch.nn.Linear(
            self.alexNet_classifier.classifier[6].out_features,
            WASHINGTON_CLASSES
        )

        for net in [self.alexNet_classifier]:
            for name, network_module in net.named_children():
                for param in network_module.parameters():
                    param.requires_grad = False

        self.alexNet_classifier.classifier[6].requires_grad = True
        # self.dropout = F.dropout()

    def forward(self, x):
        h_alex = self.alexNet_classifier(x)
        prediction = self.ensemble(F.relu(h_alex))
        return prediction


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, default=36, help='input batch size')
    parser.add_argument('--num_points', type=int, default=2500, help='input batch size')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--nepoch', type=int, default=50, help='number of epochs to train for')
    parser.add_argument('--gpu', type=str, default="", help='number of epochs to train for')
    parser.add_argument('--outf', type=str, default='cls', help='output folder')
    parser.add_argument('--model', type=str, default='', help='model path')
    return parser.parse_args()


def load_data(opt):
    model_to_use = 'not_frozen' if opt.train_pointnet else 'frozen'
    split = opt.dset
    folder = '/' + split + '/' + model_to_use + '/fc' + str(opt.nfc) + '/'

    root_weights = RESULTS_HOME + './weights' + folder
    root_data = RESULTS_HOME + './data_acc_loss' + folder

    if opt.model == '' and (os.path.exists(root_weights) or os.path.exists(root_data)):
        print('Directory Exists')
        exit()

    if not os.path.exists(root_weights):
        os.makedirs(root_weights)
        print("storing weigths in ", root_weights)
    if not os.path.exists(root_data):
        os.makedirs(root_data)
        print("storing data in ", root_data)
    Batch_size = 24
    test_dataset = bi_deco.datasets.washington.WASHINGTON_Dataset(data_dir=RESOURCES_HOME + '/dataset/' + split + '/val_db',
                                                                  train=False)
    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=Batch_size, shuffle=False, num_workers=1)
    return testdataloader


def main():
    opt = parser_args()
    opt.gpu = "3"
    model = Alexnet_only()
    if opt.gpu != "":
        model.cuda()
    print(model)
    train_loader, test_loader = bi_deco.datasets.washington.load_dataset(
        data_dir='/home/alessandrodm/tesi/dataset/', split="5", batch_size=opt.batchSize, rgb=True)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=0.007, momentum=0.9, nesterov=True)

    criterion = torch.nn.CrossEntropyLoss()
    target_Variable = torch.LongTensor(opt.batchSize)

    epoch_train_loss = []
    epochs_loss = []
    epochs_accuracy = []

    for epoch in range(opt.nepoch):
        print("EPOCH {}/{} ".format(epoch, opt.nepoch))
        model.train()

        progress_bar = tqdm.tqdm(total=len(train_loader))
        epoch_losses = collections.deque(maxlen=100)
        for step, (depths, y_target) in enumerate(train_loader, 0):
            if step > len(train_loader) / 2:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 0.001

            progress_bar.update(1)
            y_target = target_Variable.copy_(y_target)

            depths, y_target = Variable(depths), Variable(y_target)

            # depths = Variable(depths)
            depths = depths.transpose(2, 1)
            if opt.gpu != "":
                depths, y_target = depths.cuda(), y_target.cuda()
            optimizer.zero_grad()

            y_pred = model(depths)
            loss = criterion(y_pred, y_target)
            loss.backward()
            loss_ = loss.cpu().data[0]
            epoch_losses.append(loss_)
            progress_bar.set_description("training loss {}, avg {}".format(loss_, np.mean(epoch_losses)))
            optimizer.step()

        epoch_train_loss.append(sum(epoch_losses) / len(epoch_losses))
        test_samples = 0
        test_losses = []
        test_accuracies = []
        freqs = {i: 0 for i in range(51)}
        model.eval()

        progress_bar.close()
        progress_bar = tqdm.tqdm(total=len(train_loader))
        for test_step, (depths, y_target) in enumerate(test_loader):
            progress_bar.update(1)
            y_target = target_Variable.copy_(y_target)
            depths, y_target = Variable(depths), Variable(y_target)

            # depths = Variable(depths)
            depths = depths.transpose(2, 1)
            if opt.gpu != "":
                depths, y_target = depths.cuda(), y_target.cuda()

            y_pred = model(depths)

            loss = criterion(y_pred, y_target).cpu().data[0]
            pred_choice = y_pred.data.max(1)[1]  # [1] is argmax, [0] would be max

            test_losses.append(loss)
            for p in pred_choice:
                freqs[p] += 1
            accuracy = pred_choice.eq(y_target.data).cpu().sum()
            progress_bar.set_description("accuracy {}".format(np.mean(test_accuracies)))
            test_accuracies.append(accuracy)
            test_samples += 1

        print("frequencies:".format({k: v for k, v in freqs.items() if v > 0}))
        progress_bar.close()
        epochs_loss.append((sum(test_losses) / test_samples))
        epochs_accuracy.append((sum(test_accuracies) / test_samples))

        print("acc")
        print("acc", epochs_accuracy)
        print("loss", epochs_loss)

        torch.save(model.state_dict(), 'state_dicts/cls_model_{:d}.pth'.format(epoch))

    with open("statistics/stats.pkl", "w") as fout:
        pickle.dump((epoch_train_loss, epochs_loss, epochs_accuracy), fout)

    plt.plot(epoch_train_loss)
    plt.plot(epochs_loss)
    plt.plot(epochs_accuracy)
    plt.savefig("plots/metrics.png")


if __name__ == "__main__":
    main()
