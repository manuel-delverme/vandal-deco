from __future__ import print_function
import os
import argparse


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
    parser.add_argument('--nr_points', type=int, default=2500, help='input batch size')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--nepoch', type=int, default=50, help='number of epochs to train for')
    parser.add_argument('--record_experiment', type=bool, default=False, help='if true; archive code as tar')
    parser.add_argument('--size', type=int, default=224, help='fml')
    parser.add_argument('--crop_size', type=int, default=224, help='fml')
    parser.add_argument('--gpu', type=str, default="2", help='gpu bus id')
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
import matplotlib.pyplot as plt
import bi_deco.models
import bi_deco.datasets.washington
import torchvision.transforms
import torch.nn.parallel
import torch.utils.data
import os
import torch.optim
import torch.utils.data
import PIL
import subprocess
from torch.backends import cudnn
from torch.autograd import Variable

RESOURCES_HOME = "/home/alessandrodm/tesi/"
RESULTS_HOME = "/home/iodice/alessandro_results/"

# cudnn.benchmark = True
# cudnn.fastest = True  # it increase memory consumption


class Bi_Deco(torch.nn.Module):
    def __init__(self, dropout_probability=0.5, nr_points=2500, ensemble_hidden_size=2048):
        WASHINGTON_CLASSES = 51
        super(Bi_Deco, self).__init__()

        self.alexNet_classifier = bi_deco.models.AlexNet(num_classes=WASHINGTON_CLASSES, pretrained=True, )
        self.alexNet_deco = bi_deco.models.deco.DECO(is_alex_net=True, nr_points=nr_points, pretrained=True)

        self.pointNet_classifier = bi_deco.models.pointnet.PointNetClassifier(num_points=nr_points, pretrained=True,
                                                                              k=WASHINGTON_CLASSES)
        self.pointNet_deco = bi_deco.models.deco.DECO(is_alex_net=False, nr_points=nr_points, pretrained=True)

        self.dropout = torch.nn.Dropout(p=dropout_probability)

        self.ensemble_fc = torch.nn.Linear(
            self.alexNet_classifier.fc8.out_features + self.pointNet_classifier.fc3.out_features,
            ensemble_hidden_size
        )
        self.ensemble_classifier = torch.nn.Linear(
            ensemble_hidden_size,
            WASHINGTON_CLASSES
        )
        self.ensemble_classifier_no_hidden = torch.nn.Linear(
            self.alexNet_classifier.fc8.out_features + self.pointNet_classifier.fc3.out_features,
            WASHINGTON_CLASSES
        )

    def forward(self, x):
        augmented_image = self.alexNet_deco(x)
        prediction_alexNet = self.alexNet_classifier(augmented_image)

        point_cloud = self.pointNet_deco(x)
        prediction_pointNet = self.pointNet_classifier(point_cloud)

        h_concat = torch.cat((prediction_alexNet, prediction_pointNet), dim=1)

        h_dropout = self.dropout(h_concat)
        h_ensemble = self.ensemble_fc(h_dropout)
        prediction = self.ensemble_classifier(h_ensemble)
        return prediction


def main(experiment_name):
    # TODO: https://ikhlestov.github.io/pages/machine-learning/pytorch-notes/#additional-topics
    # TODO: try out different losses
    # TODO: try out differnt pooling
    # TODO: try out different losses
    # TODO: http://neupy.com/pages/cheatsheet.html
    # TODO: torch.optim.lr_scheduler.ReduceLROnPlateau

    opt = parser_args()
    # TODO: ensemble layer N -> 51 [DONE, with no dropout on ensemble]
    # TODO: ensemble layer N -> n2 -> 51  n2=2048 [DONE], 4096 [DONE]
    # TODO: classify with full_branch dropout, turning off one branch at time instead of w/p 0.5
    # TODO: try more epochs
    # TODO: try pretrained DECO1, DECO2
    # TODO: test split 0, 1,2, 3, 4,cffff

    # Data pre-processing
    # TODO: use these vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    transforms_ = [
        torchvision.transforms.Scale(int(opt.size), PIL.Image.BICUBIC),
        torchvision.transforms.RandomCrop(opt.crop_size),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5), (0.5))
    ]

    classifier = Bi_Deco(nr_points=opt.nr_points, ensemble_hidden_size=2048)
    if opt.gpu != "":
        classifier.cuda()
    print(classifier)
    train_loader, test_loader = bi_deco.datasets.washington.load_dataset(
        data_dir='/home/alessandrodm/tesi/dataset/',
        split="5",
        batch_size=opt.batch_size
    )

    crossEntropyLoss = torch.nn.CrossEntropyLoss().cuda()
    # optimizer = torch.optim.SGD(get_trainable_params(model), lr=0.007, momentum=0.9, nesterov=True)
    class_optimizer = torch.optim.SGD(get_trainable_params(classifier), lr=0.007, momentum=0.9, nesterov=True)

    target_Variable = torch.LongTensor(opt.batch_size)

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
        except OSError:
            pass

        try:
            os.mkdir('statistics/')
        except OSError:
            pass

        torch.save(classifier.state_dict(), 'state_dicts/{}cls_model_{:d}.pth'.format(experiment_name, epoch))
        with open("statistics/{}_stats{}.pkl".format(experiment_name, epoch), "w") as fout:
            pickle.dump((epoch_train_loss, epochs_test_loss, epochs_accuracy), fout)

    plt.plot(epoch_train_loss)
    plt.plot(epochs_test_loss)
    plt.plot(epochs_accuracy)
    plt.savefig("plots/metrics.png")


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
    target_Variable = torch.LongTensor(opt.batch_size)

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


def get_trainable_params(model):
    params = []
    print("training parameters:")
    for name, p in model.named_parameters():
        if p.requires_grad:
            print(name)
            params.append(p)
    return params


if __name__ == "__main__":
    import time
    timestr = time.strftime("%Y_%m_%d-%H_%M_%S")
    print("Experiment name:", timestr)
    if opt.record_experiment:
        print("archived")
        cmd = "find /home/iodice/vandal-deco/bi_deco -name '*.py' | tar -cvf run{}.tar --files-from -".format(timestr)
        subprocess.check_output(cmd, shell=True)
    main(experiment_name=timestr)
