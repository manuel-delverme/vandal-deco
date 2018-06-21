import argparse
import os
import tf_logger
import torchsummary


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
    parser.add_argument('--nr_points', type=int, default=2500, help='')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
    parser.add_argument('--nepoch', type=int, default=50, help='number of epochs to train for')
    parser.add_argument('--size', type=int, default=224, help='fml')
    parser.add_argument('--ensemble_hidden_size', type=int, default=2048)
    parser.add_argument('--crop_size', type=int, default=224, help='fml')
    parser.add_argument('--gpu', type=str, default="-1", help='gpu bus id')
    parser.add_argument('--experiment_name', type=str, default="")
    parser.add_argument('--from_scratch', action='store_true')
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
    parser.add_argument('--description', type=str, default='', help='descriptive word for the experiment')
    return parser.parse_args()


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
opt = parser_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)

from bi_deco import utils
import torch
import tqdm
import torch.nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.parallel
import os
import torch.optim
import subprocess
# import models.bi_deco as bi_deco_model
import models.bi_deco_dirty as bi_deco_model
from torch.autograd import Variable

RESOURCES_HOME = "/home/alessandrodm/tesi/"
RESULTS_HOME = "/home/iodice/alessandro_results/"


def train_bideco(experiment_name, resume_experiment=False):
    logger = tf_logger.Logger("tf_log/{}".format(experiment_name))

    classifier, crossEntropyLoss, experiment_epoch, learning_rate = load_classifier(experiment_name, logger, resume_experiment)
    learning_rate /= 10

    if opt.use_adam:
        class_optimizer = torch.optim.Adam(utils.get_trainable_params(classifier), lr=3e-4)
    else:
        class_optimizer = torch.optim.SGD(utils.get_trainable_params(classifier), lr=learning_rate, momentum=0.9,
                                          nesterov=True)

    test_loader, train_loader = utils.load_dataset(opt.split, opt.batch_size)

    last_test_accuracy = -1
    progress_bar = tqdm.tqdm(total=len(train_loader) * opt.nepoch)
    samples_seen = len(train_loader) * (experiment_epoch + 1)
    progress_bar.update(samples_seen)
    tf_step = samples_seen

    for epoch in range(experiment_epoch + 1, opt.nepoch):
        if opt.decimate_lr and epoch == 40:
            class_optimizer.param_groups[0]['lr'] = learning_rate / 10.

        classifier.train()
        progress_bar.set_description("epoch {} lr {} accuracy {}".format(0, class_optimizer.param_groups[0]['lr'], last_test_accuracy))

        for step, (inputs, labels) in enumerate(train_loader, 0):
            if opt.skip_training and step > 5:
                break
            progress_bar.update(1)
            tf_step += 1

            target_Variable = torch.LongTensor(opt.batch_size)
            labels = target_Variable.copy_(labels)

            inputs, labels = Variable(inputs), Variable(labels)
            inputs, labels = inputs.cuda(), labels.cuda()

            class_pred = classifier(inputs)
            class_loss = crossEntropyLoss(class_pred, labels)
            class_optimizer.zero_grad()
            class_loss.backward()
            class_optimizer.step()
            loss_ = class_loss.data[0]
            # FOR FUCKS SAKE
            torch.cuda.empty_cache()

            logger.scalar_summary("bi_deco/train_loss", loss_, tf_step)
            progress_bar.set_description("epoch {} lr {} accuracy {}".format(epoch, class_optimizer.param_groups[0]['lr'],
                                         last_test_accuracy))

        del inputs
        del labels
        test_accuracy, test_loss = test(crossEntropyLoss, classifier, opt, test_loader)
        last_test_accuracy = test_accuracy
        logger.scalar_summary("bi_deco/test_accuracy", test_accuracy, tf_step)

        if not opt.skip_training:
            try:
                os.mkdir('state_dicts/')
            except OSError:
                pass
            torch.save(classifier.state_dict(), 'state_dicts/{}cls_model_{:d}.pth'.format(experiment_name, epoch))


def load_classifier(experiment_name, logger, resume_experiment):
    classifier = bi_deco_model.Bi_Deco(
        nr_points=opt.nr_points,
        ensemble_hidden_size=opt.ensemble_hidden_size,
        batch_norm2d=opt.batch_norm2d,
        bound_pointnet_deco=opt.bound_pointnet_deco,
        record_pcls=opt.record_pcls,
        branch_dropout=opt.branch_dropout,
        logger=logger,
        from_scratch=opt.from_scratch,
    )
    if resume_experiment:
        checkpoint_path, experiment_epoch = utils.latest_checkpoint_path(experiment_name)
        classifier.load_state_dict(torch.load(checkpoint_path))
    else:
        experiment_epoch = -1
    assert opt.gpu != "-1"
    classifier.cuda()
    crossEntropyLoss = torch.nn.CrossEntropyLoss().cuda()
    if experiment_epoch > 40 and opt.decimate_lr:
        learning_rate = 0.0007
    else:
        learning_rate = 0.007
    return classifier, crossEntropyLoss, experiment_epoch, learning_rate


def test(CrossEntropyLoss, classifier, opt, test_loader):
    classifier.eval()
    correct = 0.0
    test_loss = 0.0
    total = 0.0

    assert test_loader.batch_size == opt.batch_size
    target_Variable = torch.LongTensor(opt.batch_size)

    for test_step, (inputs, labels) in enumerate(test_loader):
        if opt.skip_training and test_step > 5:
            break
        labels = target_Variable.copy_(labels)

        inputs, labels = inputs.cuda(), labels.cuda()
        inputs, labels = Variable(inputs, volatile=True), Variable(labels, volatile=True)

        class_pred = classifier(inputs)
        class_loss = CrossEntropyLoss(class_pred, labels)
        _, predicted = torch.max(class_pred.data, 1)
        total += labels.size(0)
        correct += predicted.eq(labels.data).cpu().sum()
        test_loss += class_loss.data[0]
        # progress_bar.set_description("accuracy {}".format(float(correct) / total))

    # progress_bar.close()
    return correct / total, test_loss / total


def main():
    if opt.experiment_name != "":
        train_bideco(experiment_name=opt.experiment_name, resume_experiment=True)
    else:
        import time
        experiment_name = time.strftime("%Y_%m_%d-%H_%M_%S")
        if opt.description != "":
            experiment_name += opt.description
        if opt.record_experiment:
            cmd = "find /home/iodice/vandal-deco/bi_deco -name '*.py' | tar -cvf run{}.tar --files-from -".format(
                experiment_name)
            subprocess.check_output(cmd, shell=True)
        train_bideco(experiment_name=experiment_name)


if __name__ == "__main__":
    main()
