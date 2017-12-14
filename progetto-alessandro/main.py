from __future__ import print_function
import tqdm

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# is this even working maybe it has to be declared earlier
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU

import argparse
import numpy as np
import models.pointnet
import models.deco
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
    parser.add_argument('--nepoch', type=int, default=25, help='number of epochs to train for')
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
    print(opt)

    model = models.bi_deco.BiDeco()
    # model = models.deco.DECO_medium_conv()

    # move the following inside BiDeco, freeze alexnet && pointnet
    # # freeze the classifier parameters
    # for layer in model.classifier.named_children():
    #     if layer[0] not in ('fc3',):
    #         for param in layer[1].parameters():
    #             param.requires_grad = False

    train_loader, test_loader = datasets.washington.load_dataset(data_dir='./dataset/', split="5", batch_size=256)

    num_classes = len(train_loader.classes)
    print('classes', num_classes)

    if opt.gpu != "":
        model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
    # optimizer = optim.Adam(model.parameters())
    # num_batch = len(train_loader) / opt.batchSize

    for epoch in range(opt.nepoch):
        progress_bar = tqdm.tqdm(total=len(train_loader))
        last_test = -1
        for step, (points, y_target) in enumerate(train_loader, 0):
            progress_bar.update(1)
            points, y_target = Variable(points), Variable(y_target[:, 0])
            points = points.transpose(2, 1)
            points, y_target = points.cuda(), y_target.cuda()
            optimizer.zero_grad()

            y_pred, _ = model(points)
            loss = F.nll_loss(y_pred, y_target)
            loss.backward()
            optimizer.step()
            pred_choice = y_pred.data.max(1)[1]
            correct = pred_choice.eq(y_target.data).cpu().sum()
            progress_bar.set_description('train loss: {} accuracy: {}'.format(
                loss.data[0], correct / float(opt.batchSize), last_test
            ))

        model.eval()
        j, (points, y_target) = enumerate(test_loader, 0).next()
        points, y_target = Variable(points), Variable(y_target[:, 0])
        points = points.transpose(2, 1)
        points, y_target = points.cuda(), y_target.cuda()
        y_pred, _ = model(points)
        pred_choice = y_pred.data.max(1)[1]
        correct = pred_choice.eq(y_target.data).cpu().sum()
        last_test = correct / float(opt.batchSize)
        torch.save(model.state_dict(), '{}/cls_model_{:d}.pth'.format(opt.outf, epoch))


if __name__ == "__main__":
    main()
