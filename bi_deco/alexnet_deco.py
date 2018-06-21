from __future__ import print_function
import argparse
import os
import tf_logger
from bi_deco import utils


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
    parser.add_argument('--nr_points', type=int, default=2500, help='')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
    parser.add_argument('--nepoch', type=int, default=50, help='number of epochs to train for')
    parser.add_argument('--size', type=int, default=224, help='fml')
    parser.add_argument('--lr', type=float, default=0.007, help='learning rate')
    parser.add_argument('--ensemble_hidden_size', type=int, default=2048)
    parser.add_argument('--crop_size', type=int, default=224, help='fml')
    parser.add_argument('--gpu', type=str, default="-1", help='gpu bus id')
    parser.add_argument('--experiment_name', type=str, default="")
    parser.add_argument('--batch_norm2d', action='store_true')
    parser.add_argument('--branch_dropout', action='store_true')
    parser.add_argument('--skip_training', action='store_true')
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
opt = parser_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)

import torch
import tqdm
import torch.nn
import torch.nn.parallel
import torch.utils.data
import bi_deco.models
import bi_deco.models.alex_net
# import bi_deco.datasets.washington
import bi_deco.datasets.washington_alexnet
import torch.nn.parallel
import torch.utils.data
import torch.nn
import os
import torch.optim
import torch.optim.lr_scheduler
import torch.utils.data
import subprocess
import torch.nn
from torch.autograd import Variable

RESOURCES_HOME = "/home/alessandrodm/tesi/"
RESULTS_HOME = "/home/iodice/alessandro_results/"
WASHINGTON_CLASSES = 51


class Alexnet_Deco(torch.nn.Module):
    def __init__(self, batch_norm2d):
        WASHINGTON_CLASSES = 51
        super(Alexnet_Deco, self).__init__()

        self.alexNet_deco = bi_deco.models.DECO_alexNet()
        # self.alexNet_deco = bi_deco.models.deco.DECO(
        #     is_alex_net=True,
        #     bound_output=False,
        #     batch_norm2d=batch_norm2d,
        # )
        self.alexNet_classifier = bi_deco.models.alex_net.AlexNet(
            num_outputs=WASHINGTON_CLASSES,
        )

    def forward(self, x):
        image = self.alexNet_deco(x)
        prediction = self.alexNet_classifier(image)
        return prediction


def train_alexnet(experiment_name, resume_experiment=False):
    prefix = "only_alexnet"
    print("loading classifier")
    logger = tf_logger.Logger("tf_log/{}_{}".format(prefix, experiment_name))
    classifier = Alexnet_Deco(
        batch_norm2d=True
    )
    experiment_epoch = -1

    if resume_experiment:
        raise NotImplementedError("not implemented for pointnet")

    if opt.gpu != "-1":
        print("loading classifier in GPU")
        classifier.cuda()

    # train_loader, test_loader = bi_deco.datasets.washington.load_dataset(
    train_loader, test_loader = bi_deco.datasets.washington_alexnet.load_dataset(
        data_dir='/scratch/dataset/',
        split=opt.split,
        batch_size=opt.batch_size,
        rgb=True,
    )

    print("loss and optimizer")
    crossEntropyLoss = torch.nn.CrossEntropyLoss().cuda()
    if experiment_epoch > 40 and opt.decimate_lr:
        learning_rate = opt.lr / 10
    else:
        learning_rate = opt.lr

    if opt.use_adam:
        class_optimizer = torch.optim.Adam(utils.get_trainable_params(classifier), lr=3e-4)
    else:
        class_optimizer = torch.optim.SGD(
            utils.get_trainable_params(classifier),
            lr=learning_rate,
            momentum=0.9,
            nesterov=True,
            weight_decay=0.0001,
        )
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     class_optimizer, 'min',
        #     patience=3,
        #     verbose=True,
        # )

    last_test_accuracy = -1
    for epoch in range(experiment_epoch + 1, opt.nepoch):
        if opt.decimate_lr and epoch == 40:
            class_optimizer.param_groups[0]['lr'] = learning_rate / 10.

        classifier.train()
        progress_bar = tqdm.tqdm(total=len(train_loader) * (50 - experiment_epoch - 1))
        criterion = torch.nn.CrossEntropyLoss()

        for step, (inputs, labels) in enumerate(train_loader, 0):
            if opt.skip_training and step > 5:
                break
            progress_bar.update(1)

            images = Variable(inputs).cuda()
            labels = Variable(labels).cuda()
            labels = labels.long()
            class_optimizer.zero_grad()
            outputs_0 = classifier(images)
            loss_0 = criterion(outputs_0, labels)
            loss_0.backward()
            class_optimizer.step()

            # target_Variable = torch.LongTensor(opt.batch_size)
            # labels = target_Variable.copy_(labels)

            # inputs, labels = Variable(inputs), Variable(labels)
            # if opt.gpu != "-1":
            #     inputs, labels = inputs.cuda(), labels.cuda()
            # class_pred = classifier(inputs)
            # class_loss = crossEntropyLoss(class_pred, labels)
            # class_optimizer.zero_grad()
            # class_loss.backward()
            # class_optimizer.step()
            # loss_ = class_loss.data[0]

            logger.scalar_summary("loss/train_loss", loss_0, step + opt.nepoch * epoch)
            progress_bar.set_description("epoch {} lr {}".format(epoch, class_optimizer.param_groups[0]['lr']),
                                         last_test_accuracy)

        del inputs
        del labels
        test_accuracy, test_loss = test(crossEntropyLoss, classifier, opt, test_loader)
        last_test_accuracy = test_accuracy

        logger.scalar_summary("loss/test_loss", test_loss, step + opt.nepoch * epoch)
        logger.scalar_summary("loss/test_accuracy", test_accuracy, step + opt.nepoch * epoch)
        # if scheduler:
        #    scheduler.step(test_loss)

        if not opt.skip_training:
            try:
                os.mkdir('state_dicts/')
            except OSError:
                pass
            torch.save(classifier.state_dict(),
                       'state_dicts/{}{}cls_model_{:d}.pth'.format(prefix, experiment_name, epoch))


def test(CrossEntropyLoss, classifier, opt, test_loader):
    classifier.eval()
    correct = 0.0
    correct_0 = 0.0
    total_0 = 0.0
    test_loss = 0.0
    nr_batches = 0.0
    total = 0.0

    progress_bar = tqdm.tqdm(total=len(test_loader))
    target_Variable = torch.LongTensor(opt.batch_size)
    criterion = torch.nn.CrossEntropyLoss()

    for test_step, (inputs, labels) in enumerate(test_loader):
        if opt.skip_training and test_step > 5:
            break
        images = Variable(inputs, volatile=True).cuda()
        labels = Variable(labels, volatile=True).cuda()
        labels = labels.long()
        outputs_0 = classifier(images)
        loss_0 = criterion(outputs_0, labels)
        predicted_0 = outputs_0.data.max(1)[1]
        correct_0 += predicted_0.eq(labels.data).cpu().sum()
        total_0 += labels.size(0)

    accuracy_0 = 1. * correct_0 / total_0

        # labels = target_Variable.copy_(labels)
        # progress_bar.update(1)
        # if opt.gpu != "":
        #     inputs, labels = inputs.cuda(), labels.cuda()
        # inputs, labels = Variable(inputs, volatile=True), Variable(labels, volatile=True)
#
#         class_pred = classifier(inputs)
#         class_loss = CrossEntropyLoss(class_pred, labels)
#         _, predicted = torch.max(class_pred.data, 1)
#         nr_batches += 1
#         total += labels.size(0)
#         correct += predicted.eq(labels.data).cpu().sum()
#         test_loss += class_loss.data[0]
#         progress_bar.set_description("accuracy {}".format(correct / total))

    # progress_bar.close()
    # return correct / total, test_loss / nr_batches
    return accuracy_0, -1


def main():
    print(opt)

    if opt.experiment_name != "":
        train_alexnet(experiment_name=opt.experiment_name, resume_experiment=True)
    else:
        import time
        experiment_name = time.strftime("%Y_%m_%d-%H_%M_%S")
        print("Experiment name:", experiment_name)
        if opt.record_experiment:
            print("archived")
            cmd = "find /home/iodice/vandal-deco/bi_deco -name '*.py' | tar -cvf run{}.tar --files-from -".format(
                experiment_name)
            subprocess.check_output(cmd, shell=True)
        train_alexnet(experiment_name=experiment_name)


if __name__ == "__main__":
    main()
