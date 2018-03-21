from __future__ import print_function

import argparse

import os
import utils


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
    parser.add_argument('--nr_points', type=int, default=2500, help='input batch size')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--nepoch', type=int, default=50, help='number of epochs to train for')
    parser.add_argument('--size', type=int, default=224, help='fml')
    parser.add_argument('--ensemble_hidden_size', type=int, default=2048)
    parser.add_argument('--crop_size', type=int, default=224, help='fml')
    parser.add_argument('--gpu', type=str, default="-1", help='gpu bus id')
    parser.add_argument('--batch_norm2d', action='store_true')
    parser.add_argument('--bound_pointnet_deco', action='store_true')
    parser.add_argument('--record_pcls', action='store_true')
    parser.add_argument('--split', type=str, default="5", help='dataset split to test on')
    parser.add_argument('--use_adam', action='store_true')
    parser.add_argument('--decimate_lr', action='store_true')
    parser.add_argument('--record_experiment', action='store_true')
    parser.add_argument("-q", action="store_false", dest="verbose")

    parser.add_argument('--outf', type=str, default='cls', help='output folder')
    parser.add_argument('--model', type=str, default='', help='model path')
    return parser.parse_args()


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# is this even working maybe it has to be declared earlier
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # CPU
opt = parser_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)

import torch
import tqdm
import torch.nn
import torch.nn.parallel
import torch.utils.data
import pickle
import collections
import numpy as np
import bi_deco.models
import bi_deco.datasets.washington
import torch.nn.parallel
import torch.utils.data
import os
import torch.optim
import torch.utils.data
import subprocess
import models.bi_deco
from torch.autograd import Variable

RESOURCES_HOME = "/home/alessandrodm/tesi/"
RESULTS_HOME = "/home/iodice/alessandro_results/"


def main(experiment_name):
    opt = parser_args()

    classifier = models.bi_deco.Bi_Deco(
        nr_points=opt.nr_points,
        ensemble_hidden_size=opt.ensemble_hidden_size,
        batch_norm2d=opt.batch_norm2d,
        bound_pointnet_deco=opt.bound_pointnet_deco,
        record_pcls=opt.record_pcls,
    )
    if opt.gpu != "-1":
        classifier.cuda()
    # print(classifier)

    train_loader, test_loader = bi_deco.datasets.washington.load_dataset(
        data_dir='/scratch/dataset/',
        split=opt.split,
        batch_size=opt.batch_size
    )

    crossEntropyLoss = torch.nn.CrossEntropyLoss().cuda()
    if opt.use_adam:
        class_optimizer = torch.optim.Adam(utils.get_trainable_params(classifier), lr=3e-4)
    else:
        class_optimizer = torch.optim.SGD(utils.get_trainable_params(classifier), lr=0.007, momentum=0.9, nesterov=True)

    target_Variable = torch.LongTensor(opt.batch_size)

    epoch_train_loss = []
    epochs_test_loss = []
    epochs_accuracy = []
    for epoch in range(opt.nepoch):
        if opt.decimate_lr and epoch == 40:
            class_optimizer.param_groups[0]['lr'] = opt.lr / 10

        print("EPOCH {}/{} ".format(epoch, opt.nepoch))
        classifier.train()
        epoch_losses = collections.deque(maxlen=100)
        progress_bar = tqdm.tqdm(total=len(train_loader))

        for step, (inputs, labels) in enumerate(train_loader, 0):
            progress_bar.update(1)
            labels = target_Variable.copy_(labels)
            inputs, labels = Variable(inputs), Variable(labels)
            if opt.gpu != "-1":
                inputs, labels = inputs.cuda(), labels.cuda()

            class_pred = classifier(inputs)
            class_loss = crossEntropyLoss(class_pred, labels)

            class_optimizer.zero_grad()
            class_loss.backward()
            class_optimizer.step()

            loss_ = class_loss.data[0]

            epoch_losses.append(loss_)
            progress_bar.set_description("avg {}".format(np.round(np.mean(epoch_losses), 2)))

        del inputs
        del labels
        del class_loss
        torch.cuda.empty_cache()
        epoch_train_loss.append(sum(epoch_losses) / len(epoch_losses))

        test_accuracy, test_loss = test(crossEntropyLoss, classifier, opt, test_loader)
        torch.cuda.empty_cache()
        epochs_test_loss.append(test_loss)
        epochs_accuracy.append(test_accuracy)

        print("acc")
        print("acc", epochs_accuracy)
        print("loss", epochs_test_loss)

        try:
            os.mkdir('state_dicts/')
        except OSError:
            pass

        try:
            os.mkdir('statistics/')
        except OSError:
            pass

        torch.save(classifier.state_dict(), 'state_dicts/{}cls_model_{:d}.pth'.format(experiment_name, epoch))
        with open("statistics/{}_stats{}.pkl".format(experiment_name, epoch), "w") as fout:
            pickle.dump((epoch_train_loss, epochs_test_loss, epochs_accuracy), fout)


def test(CrossEntropyLoss, classifier, opt, test_loader):
    classifier.eval()
    correct = 0.0
    test_loss = 0.0
    total = 0.0

    progress_bar = tqdm.tqdm(total=len(test_loader))
    target_Variable = torch.LongTensor(opt.batch_size)

    for test_step, (inputs, labels) in enumerate(test_loader):
        labels = target_Variable.copy_(labels)
        progress_bar.update(1)
        if opt.gpu != "":
            inputs, labels = inputs.cuda(), labels.cuda()
        inputs, labels = Variable(inputs, volatile=True), Variable(labels, volatile=True)

        class_pred = classifier(inputs)
        class_loss = CrossEntropyLoss(class_pred, labels)
        _, predicted = torch.max(class_pred.data, 1)
        total += labels.size(0)
        correct += predicted.eq(labels.data).cpu().sum()
        test_loss += class_loss.data[0]
        progress_bar.set_description("accuracy {}".format(correct / total))

    progress_bar.close()
    return correct / total, test_loss / total


if __name__ == "__main__":
    import time
    timestr = time.strftime("%Y_%m_%d-%H_%M_%S")
    print("Experiment name:", timestr)
    if opt.record_experiment:
        print("archived")
        cmd = "find /home/iodice/vandal-deco/bi_deco -name '*.py' | tar -cvf run{}.tar --files-from -".format(timestr)
        subprocess.check_output(cmd, shell=True)
    main(experiment_name=timestr)
