import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import math
# import tf_logger
import itertools
import torch.utils.data as data
import torch.nn.parallel
import scipy.ndimage as nd
import torchvision as vision
from torch.autograd import Variable

import h5py

from classDecoAlex_onlyDepth import *
from PIL import Image
import os.path
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--gpu', type=int, default=0, help='identification number of gpu')
parser.add_argument('--batchSize', type=int, default=8, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--nepoch', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--model', type=str, default='/home/iodice/vandal-deco/working_alexnet_weights_best_735.pkl', help='model path')
parser.add_argument('--dropout', type=float, default=0.0, help='percentage of neurons set to zero on FC1')
parser.add_argument('--wd', type=float, default=0.0001, help='weight_decay, default to 0')
parser.add_argument('--softmax', type=bool, default=True, help='enable or disable softmax at DECO output')
parser.add_argument('--test_epochs', type=int, default=10, help='run testing phase every test_epochs ')
parser.add_argument('--solver', type=str, default='Nesterov',
                    help='model solver: Adam, SGD or Nesterov. For nesterov is suggested: train_augmentation.py --nepoch')
parser.add_argument('--lr', type=float, default=0.007,
                    help='starting learning rate.')  # quanto impara la rete a ogni passo
parser.add_argument('--lr_gamma', type=float, default=0.1,
                    help='learning rate decrease factor.Suggested 1.0 for Adam, 0.1 per SGD/Nesterov')
parser.add_argument('--dset', type=str, help='dataset folder: JHUIT or split0 split1 etc')
parser.add_argument('--path', type=str, help='dataset folder')
opt = parser.parse_args()

freezed = 'freezed'
split = "split5"


class Dataset_depth(data.Dataset):
    def __init__(self, data_dir, image_size=256, train=True):
        self.image_size = image_size  # si crea le proprieta dell oggetto
        self.data_dir = data_dir
        self.train = train

        file_path = os.path.join(self.data_dir, '0.h5')
        self.washington_data = h5py.File(file_path)
        if self.train:
            self.train_data = self.washington_data['data']
            self.train_labels = self.washington_data['label']
        else:
            self.test_data = self.washington_data['data']
            self.test_labels = self.washington_data['label']

        # self.data = ReadLmdb2(data_dir)
        self.num_classes = sum(1 for line in open(os.path.join(self.data_dir, '../labels.txt')))
        self.length = len(self.washington_data['data'])
        mean_file_path = os.path.join(self.data_dir, '../mean.jpg')
        mean_image = nd.imread(mean_file_path) / 255.0
        self.mean = mean_image.mean()  # salva come proprieta dell oggetto washington la me$
        print("mean pixel value image: %.6f" % self.mean)

        self.rc = vision.transforms.RandomCrop(
            [228, 228])  # crea delle varibili per ogni metodo per richiamarle piu facilmente
        self.cc = vision.transforms.CenterCrop(
            [228, 228])  # randomcop taglia l immagine casualmente nelle dimensioni che gli ho passato
        self.resize = vision.transforms.Scale([228, 228])
        self.toTensor = vision.transforms.ToTensor()
        self.toPIL = vision.transforms.ToPILImage()
        self.flip = vision.transforms.RandomHorizontalFlip()

    def __getitem__(self,
                    index):  # restituisce l immagine in posizione index che gli passo come parametro e lo passa modificato
        if self.train:
            data, label = self.train_data[index], self.train_labels[index]
            data = Image.fromarray(data[0], mode='I')
            data = self.rc(data)  # cc is center crop, rc is randomcrop
            data = self.flip(data)
        else:
            data, label = self.test_data[index], self.test_labels[index]
            data = Image.fromarray(data[0], mode='I')
            data = self.cc(data)  # center crop perche nel tet voglio il centro dell immagine

        # rnd_shift = random.randint(-20,20)
        # data = ImageChops.offset(data,rnd_shift)
        data = self.toTensor(data)
        data = data.float()
        data -= self.mean
        return data, label

    def __len__(self):
        if self.train:
            return self.train_data.shape[0]
        else:
            return self.test_data.shape[0]


correct_train_0 = 0
total_train_0 = 0
correct_0 = 0
total_0 = 0

Batch_size = 32

dataset = Dataset_depth(data_dir='/scratch/dataset/split5/train_db/', train=True)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=Batch_size, shuffle=True, num_workers=1)

test_dataset = Dataset_depth(data_dir='/scratch/dataset/split5/val_db/', train=False)
testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=Batch_size, shuffle=False, num_workers=1)

num_classes = test_dataset.num_classes
print('classes', num_classes)

model_0 = DecoAlexNet(num_classes=num_classes).cuda()

s = nn.Softmax()
criterion = nn.CrossEntropyLoss()
lr = opt.lr
optimizer_0 = torch.optim.SGD(params=parameters_0, lr=lr, weight_decay=opt.wd, momentum=0.9, nesterov=True)

# Training
num_epoch = opt.nepoch

import glob
for model in glob.glob("epoch*freezed.pkl"):
if opt.model:
    model_0.load_state_dict(torch.load(opt.model))
    print("LOADING MODEL SNAPSHOT")

import time
experiment_name = time.strftime("%Y_%m_%d-%H_%M_%S")
prefix = "working_alexnet"
logger = tf_logger.Logger("tf_log/{}_{}".format(prefix, experiment_name))
tensorboard_step = 0


def train(epoch, lr):
    global tensorboard_step
    global parameters_0
    global optimizer_0
    global correct_train_0
    global total_train_0
    correct_train_0 = 0.0
    total_train_0 = 0.0
    model_0.train()
    for i, (images, labels) in enumerate(dataloader):
        tensorboard_step += 1

        images = Variable(images).cuda()  # trasforma in variabili di cuda
        labels = Variable(labels).cuda()
        labels = labels.long()
        # Forward + Backward + Optimize
        optimizer_0.zero_grad()  # pulisce i gradienti di tutte le variabili ottimizzate
        outputs_0 = model_0(images)
        loss_0 = criterion(outputs_0, labels)
        logger.scalar_summary("train_loss", loss_0, tensorboard_step)

        loss_0.backward()
        optimizer_0.step()
        predicted_0 = outputs_0.data.max(1)[1]

        correct_train_0 += predicted_0.eq(labels.data).cpu().sum()  # si conta quante sono corrette
        total_train_0 += labels.size(0)

        # correct_train += (predicted == labels).sum()
        accuracy_train_0 = 1. * correct_train_0 / total_train_0

        # acc_0.write(str(accuracy_train_0) + '\n')
        # loss_f_0.write(str(loss_0.data[0]) + '\n')

        if (i + 1) % 250 == 0:
            print("Epoch [%d/%d], Iter [%d]  Loss_0: %.6f Accuracy_0: %.6f  %s %s"
                  % (epoch + 1, num_epoch, i + 1, loss_0.data[0], accuracy_train_0, split, freezed))
            # res = "Epoch [" + str(epoch + 1) + '/' + str(num_epoch) + " Iter " + str(i + 1) + " Loss_0: " + str(
            #    loss_0.data[0]) + " Accuracy_0: " + str(accuracy_train_0) + " " + split + " " + freezed + "\n"
            # resume_file.write(res)

            # Save the Model
    root_weights = "./"
    filesave = root_weights + 'epoch' + str(epoch) + freezed + '.pkl'
    torch.save(model_0.state_dict(), filesave)


def test(epoch):
    global correct_0
    global total_0
    global num_epoch
    global tensorboard_step
    tensorboard_step += 1
    correct_0 = 0.0
    total_0 = 0.0
    model_0.eval()  # ; testdataloader.init_epoch()

    for i, (images, labels) in enumerate(testdataloader):
        images = Variable(images, volatile=True).cuda()
        labels = Variable(labels, volatile=True).cuda()
        labels = labels.long()
        outputs_0 = model_0(images)
        loss_0 = criterion(outputs_0, labels)
        predicted_0 = outputs_0.data.max(1)[1]
        correct_0 += predicted_0.eq(labels.data).cpu().sum()
        total_0 += labels.size(0)
        # loss_f_test_0.write(str(loss_0.data[0]) + '\n')

    accuracy_0 = 1. * correct_0 / total_0
    logger.scalar_summary("accuracy", accuracy_0, tensorboard_step)
    print("test accuracy", accuracy_0)

    # acc_test_0.write(str(accuracy_0) + '\n')
    # acc_test_0.flush()


# training/testing loop
test(0)
# for epoch in range(num_epoch):  # fa il test ogni tot epoche
#     train(epoch, lr)
#     test(epoch)
#
#     # reducing learning rate every num_epoch/2 epochs:
#     # if math.fmod(epoch + 1, num_epoch / 3) == 0:
#     if epoch == 40:
#         lr = lr * 0.1
#         # lr = lr * opt.lr_gamma  # lr_gamma usually is 0.1
#        print("new learning rate: %.6f" % lr)
