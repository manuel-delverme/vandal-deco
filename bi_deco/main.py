from __future__ import print_function
import os
import argparse

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, default=8, help='input batch size')
    parser.add_argument('--nr_points', type=int, default=2500, help='input batch size')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--nepoch', type=int, default=50, help='number of epochs to train for')
    parser.add_argument('--gpu', type=str, default=2, help='number of epochs to train for')
    parser.add_argument('--outf', type=str, default='cls', help='output folder')
    parser.add_argument('--model', type=str, default='', help='model path')
    return parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# is this even working maybe it has to be declared earlier
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # CPU
opt = parser_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)

import tqdm
import utils
import pickle
import collections
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


def test(CrossEntropyLoss, classifier, opt, test_loader):
    classifier.eval()
    correct = 0.0
    test_loss = 0.0
    total = 0.0
    counter = 0.0

    # test_samples = 0
    # test_losses = []
    # test_accuracies = []
    # freqs = {i: 0 for i in range(51)}
    progress_bar = tqdm.tqdm(total=len(test_loader))
    target_Variable = torch.LongTensor(opt.batchSize)

    for test_step, (inputs, labels) in enumerate(test_loader):
        labels = target_Variable.copy_(labels)
        progress_bar.update(1)
        if opt.gpu != "":
            inputs, labels = inputs.cuda(), labels.cuda()

        # labels = target_Variable.copy_(labels)
        inputs, labels = Variable(inputs), Variable(labels)
        # inputs = torch.cat([inputs, inputs, inputs], 1)

        class_pred = classifier(inputs)
        class_loss = CrossEntropyLoss(class_pred, labels)
        _, predicted = torch.max(class_pred.data, 1)
        total += labels.size(0)
        correct += predicted.eq(labels.data).cpu().sum()
        # for yi_pred, yi_target in zip(predicted, labels.data.cpu()):
        #     freqs[yi_pred] += 1

        # accuracy = pred_choice.eq(y_target.data).cpu().sum()
        test_loss += class_loss.data[0]
        # test_accuracies.append(corrects / float(len(predicted)))
        progress_bar.set_description("accuracy {}".format(correct / total))
    # print("frequencies:".format({k: v for k, v in freqs.items() if v > 0}))
    progress_bar.close()
    return correct / total, test_loss / total


def main():
    opt = parser_args()

    classifier = models.bi_deco.BiDeco(nr_points=2500, dropout_probability=0.5)
    if opt.gpu != "":
        classifier.cuda()
    print(classifier)

    train_loader, test_loader = datasets.washington.load_dataset(
        data_dir='/home/alessandrodm/tesi/dataset/',
        split="5",
        batch_size=opt.batchSize
    )

    crossEntropyLoss = torch.nn.CrossEntropyLoss().cuda()
    class_optimizer = torch.optim.SGD(utils.get_trainable_params(classifier), lr=0.007, momentum=0.9, nesterov=True)

    target_Variable = torch.LongTensor(opt.batchSize)

    epoch_train_loss = []
    epochs_test_loss = []
    epochs_accuracy = []
    for epoch in range(opt.nepoch):
        print("EPOCH {}/{} ".format(epoch, opt.nepoch))
        classifier.train()
        epoch_losses = collections.deque(maxlen=100)
        progress_bar = tqdm.tqdm(total=len(train_loader))

        for step, (inputs, labels) in enumerate(train_loader, 0):
            progress_bar.update(1)
            # if step > 5:
            #     break
            labels = target_Variable.copy_(labels)
            inputs, labels = Variable(inputs), Variable(labels)
            if opt.gpu != "":
                inputs, labels = inputs.cuda(), labels.cuda()
            # inputs = torch.cat([inputs, inputs, inputs], 1)

            class_pred = classifier(inputs)
            class_loss = crossEntropyLoss(class_pred, labels)

            class_optimizer.zero_grad()
            class_loss.backward()
            class_optimizer.step()

            loss_ = class_loss.data[0]

            epoch_losses.append(loss_)
            progress_bar.set_description("avg {}".format(np.round(np.mean(epoch_losses), 2)))

        epoch_train_loss.append(sum(epoch_losses) / len(epoch_losses))

        test_accuracy, test_loss = test(crossEntropyLoss, classifier, opt, test_loader)
        epochs_test_loss.append(test_loss)
        epochs_accuracy.append(test_accuracy)

        print("acc")
        print("acc", epochs_accuracy)
        print("loss", epochs_test_loss)

        try:
            os.mkdir('state_dicts/')
        except:
            pass
        torch.save(classifier.state_dict(), 'state_dicts/cls_model_{:d}.pth'.format(epoch))

    with open("statistics/stats.pkl", "w") as fout:
        pickle.dump((epoch_train_loss, epochs_test_loss, epochs_accuracy), fout)

    plt.plot(epoch_train_loss)
    plt.plot(epochs_test_loss)
    plt.plot(epochs_accuracy)
    plt.savefig("plots/metrics.png")


if __name__ == "__main__":
    main()
