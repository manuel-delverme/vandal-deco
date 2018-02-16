from __future__ import print_function
import tqdm

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# is this even working maybe it has to be declared earlier
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # CPU

import pickle
import collections
import argparse
import numpy as np
import matplotlib.pyplot as plt
import models.pointnet
import models.deco_old
import models.bi_deco
import datasets.washington
import torch
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
from torch.nn.modules.module import _addindent
import argparse
import os
import random
import torch.optim as optim
import torch.utils.data
import torch.nn
from torch.autograd import Variable
import torch.nn.functional as F

# RESOURCES_HOME = "/home/iodice/vandal-deco/progetto-alessandro/tesi/tesi/"
RESOURCES_HOME = "/home/alessandrodm/tesi/"
RESULTS_HOME = "/home/iodice/alessandro_results/"
CLASSIFIER_WEIGHTS = '/home/alessandrodm/tesi/pointnet_weights/cls/cls_model_24.pth'
BLUEIZE = lambda x: '\033[94m' + x + '\033[0m'


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
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
    test_dataset = datasets.washington.WASHINGTON_Dataset(data_dir=RESOURCES_HOME + '/dataset/' + split + '/val_db',
                                                          train=False)
    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=Batch_size, shuffle=False, num_workers=1)
    return testdataloader


def load_pointnet(model_path):
    try:
        state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
        pointNet.load_state_dict(state_dict)
    except IOError:
        model = train_pointnet()
        model.save_state_dict(model_path)
    return model


def main():
    opt = parser_args()

    # ############ ALSO CHANGE ############
    # os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # CPU
    opt.gpu = "3"
    # ############ ALSO CHANGE ############

    print(opt)

    model = models.bi_deco.BiDeco()
    # model = models.deco.DECO_medium_conv()

    # move the following inside BiDeco, freeze alexnet && pointnet
    # # freeze the classifier parameters
    # for layer in model.classifier.named_children():
    #     if layer[0] not in ('fc3',):
    #         for param in layer[1].parameters():
    #             param.requires_grad = False

    BATCH_SIZE = 8
    import torchvision
    pipeline = [
        torchvision.transforms.RandomCrop,  # +5% accuracy on domain transfer
        # TODO: add more preprocessing
    ]
    pipeline = None
    train_loader, test_loader = datasets.washington.load_dataset(data_dir='/home/alessandrodm/tesi/dataset/', split="5",
                                                                 batch_size=BATCH_SIZE, preprocess=pipeline)

    if opt.gpu != "":
        model.cuda()

    # todo use 0.001 as lr
    # optimizer = optim.SGD(model.parameters(), lr=0.007, momentum=0.9, nesterov=True)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=0.007, momentum=0.9, nesterov=True)

    # optimizer = optim.Adam(model.parameters())
    # num_batch = len(train_loader) / opt.batchSize

    criterion = torch.nn.CrossEntropyLoss()
    target_Variable = torch.LongTensor(BATCH_SIZE)

    epoch_train_loss = []
    epochs_loss = []
    epochs_accuracy = []

    for epoch in range(opt.nepoch):
        print("EPOCH {}/{} ".format(epoch, opt.nepoch))
        model.train()

        progress_bar = tqdm.tqdm(total=len(train_loader))
        epoch_losses = collections.deque(maxlen=100)
        for step, (depths, y_target) in enumerate(train_loader, 0):
            if step > 200:
                break

            if step > len(train_loader)/2:
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

        epoch_train_loss.append(sum(epoch_losses)/len(epoch_losses))
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
