from __future__ import print_function
import os
import argparse
import math
import datetime
import torch.utils.data as data
import itertools
import torch.nn.parallel
import torch.utils.data
import torchvision as vision
import h5py
from DECO_gpu_fix02 import *
import torch.utils.data as data
from PIL import Image
from PIL import ImageChops
import os.path

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
RESOURCES_HOME = "/home/iodice/vandal-deco/progetto-alessandro/tesi/tesi/"


class WASHINGTON_Dataset(data.Dataset):
    def __init__(self, data_dir, image_size=256, train=True):
        self.image_size = image_size
        self.data_dir = data_dir
        self.train = train

        file_path = os.path.join(self.data_dir, '0.h5')
        self.washington_data = h5py.File(file_path)
        mean_file_path = os.path.join(self.data_dir, '../mean.jpg')
        mean_image = np.asarray(Image.open(mean_file_path).convert('I'))
        self.mean = mean_image.mean()
        print("mean pixel value: %.6f" % self.mean)
        if self.train:
            self.train_data = self.washington_data['data']
            self.train_labels = self.washington_data['label']
        else:
            self.test_data = self.washington_data['data']
            self.test_labels = self.washington_data['label']
        self.rc = vision.transforms.RandomCrop([224, 224])
        self.cc = vision.transforms.CenterCrop([224, 224])
        self.resize = vision.transforms.Scale([224, 224])
        self.toTensor = vision.transforms.ToTensor()
        self.toPIL = vision.transforms.ToPILImage()
        self.flip = vision.transforms.RandomHorizontalFlip()

    def __getitem__(self, index):
        if self.train:
            data_w, label = self.train_data[index], self.train_labels[index]
            data_w = Image.fromarray(data_w[0], mode='I')
            data_w = self.rc(data_w)  # cc is center crop, rc is randomcrop
            data_w = self.flip(data_w)
        else:
            data_w, label = self.test_data[index], self.test_labels[index]
            data_w = Image.fromarray(data_w[0], mode='I')
            data_w = self.cc(data_w)
        rnd_shift = random.randint(-20, 20)
        data_w = ImageChops.offset(data_w, rnd_shift)
        data_w = self.toTensor(data_w)
        data_w = data_w.float()
        data_w -= self.mean
        return data_w, label

    def __len__(self):
        if self.train:
            return self.train_data.shape[0]
        else:
            return self.test_data.shape[0]


def train(model_0, dataloader, criterion, root_weights, m, num_epoch, epoch, lr):
    global parameters_0
    global optimizer_0
    global correct_train_0
    global total_train_0
    global matrix_train
    correct_train_0 = 0.0
    total_train_0 = 0.0
    model_0.train()
    for i, (images, labels) in enumerate(dataloader):
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
        labels = labels.long()
        # Forward + Backward + Optimize
        optimizer_0.zero_grad()
        outputs_0, _ = model_0(images)
        loss_0 = criterion(m(outputs_0), labels)
        # loss = F.nll_loss(outputs, labels)
        loss_0.backward()
        optimizer_0.step()
        predicted_0 = outputs_0.data.max(1)[1]

        correct_train_0 += predicted_0.eq(labels.data).cpu().sum()
        total_train_0 += labels.size(0)

        # correct_train += (predicted == labels).sum()
        accuracy_train_0 = 1. * correct_train_0 / total_train_0

        acc_0.write(str(accuracy_train_0) + '\n')
        loss_f_0.write(str(loss_0.data[0]) + '\n')

        if (i + 1) % 100 == 0:
            print("Epoch [%d/%d], Iter [%d]  Loss_0: %.6f Accuracy_0: %.6f  %s %s"
                  % (epoch + 1, num_epoch, i + 1, loss_0.data[0], accuracy_train_0, split, frozen))
    if epoch < num_epoch - 10:
        if epoch % 10 == 0:
            filesave = root_weights + 'epoch' + str(epoch) + opt.arch + opt.solver + frozen + '.pkl'
            torch.save(model_0.state_dict(), filesave)
    else:
        filesave = root_weights + 'epoch' + str(epoch) + opt.arch + opt.solver + frozen + '.pkl'
        torch.save(model_0.state_dict(), filesave)


        # Decaying Learning Rate

        #    if (epoch+1) % 20 == 0:
        #        lr /= 3
        #    optimizer_0 = torch.optim.Adam(params=parameters_0, lr=lr,weight_decay=opt.wd)

        # Save the Model


# Test

def test(model_0, testdataloader, criterion, m, num_epoch, epoch, loss_f_test_0, s, labels_p,  acc_test_0):
    correct_0 = 0.0
    total_0 = 0.0
    model_0.eval()  # ; testdataloader.init_epoch()

    for i, (images, labels) in enumerate(testdataloader):
        images = Variable(images, volatile=True).cuda()
        labels = Variable(labels, volatile=True).cuda()
        labels = labels.long()
        outputs_0, _ = model_0(images)

        loss_0 = criterion(m(outputs_0), labels)
        predicted_0 = outputs_0.data.max(1)[1]
        correct_0 += predicted_0.eq(labels.data).cpu().sum()
        total_0 += labels.size(0)
        loss_f_test_0.write(str(loss_0.data[0]) + '\n')

        if epoch == (num_epoch):
            outputs_np = s(outputs_0)
            outputs_np = outputs_np.cpu()
            outputs_np = outputs_np.data.numpy()
            # print(matrix_test.shape)
            matrix_test = np.append([matrix_test], [outputs_np])
            # matrix_train = matrix_train.view(matrix_train.size/51,51)
            matrix_test = matrix_test.reshape(-1, 51)
            # print('Max='+str(outputs_np.max())+' Min='+str(outputs_np.min())+' Sum='+str(outputs_np.sum()))

            labels_image = str(labels)
            index = labels_image.index('[')
            labels_image = labels_image[index - 3:index - 1]
            labels_image = list[int(labels_image)]

            labels_image = str(labels)
            index = labels_image.index('[')
            labels_image = labels_image[index - 3:index - 1]
            labels_image = list[int(labels_image)]

            labels_p.write(str(labels_image) + '\n')
            # correct_train += (predicted == labels).sum()

    accuracy_0 = 1. * correct_0 / total_0
    print("Test %d  Accuracy_0: %.6f"
          % (epoch, accuracy_0))

    acc_test_0.write(str(accuracy_0) + '\n')
    acc_test_0.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='identification number of gpu')
    parser.add_argument('--nfc', type=int, default=4096, help='number of fc2 s neurons')
    parser.add_argument('--k', type=int, default=1, help='not frozen = 1')
    parser.add_argument('--batchSize', type=int, default=8, help='input batch size')
    parser.add_argument('--num_points', type=int, default=2500, help='input batch size')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--nepoch', type=int, default=60, help='number of epochs to train for')
    parser.add_argument('--outf', type=str, default='cls', help='output folder')
    parser.add_argument('--model', type=str, default='', help='model path')
    parser.add_argument('--dropout', type=float, default=0.0, help='percentage of neurons set to zero on FC1')
    parser.add_argument('--wd', type=float, default=0.0, help='weight_decay, default to 0')
    parser.add_argument('--arch', type=str, default='DECO_medium_conv',
                        help='model architecture: DECO, DECO_medium_conv, DECO_heavy_conv')
    parser.add_argument('--softmax', type=bool, default=True, help='enable or disable softmax at DECO output')
    parser.add_argument('--test_epochs', type=int, default=10, help='run testing phase every test_epochs ')
    parser.add_argument('--solver', type=str, default='Adam',
                        help='model solver: Adam, SGD or Nesterov. For nesterov is suggested: train_augmentation.py --nepoch 120 --solver Nesterov --lr 0.01  --lr_gamma 0.1')
    parser.add_argument('--lr', type=float, default=0.001, help='starting learning rate.')
    parser.add_argument('--lr_gamma', type=float, default=1.0,
                        help='learning rate decrease factor.Suggested 1.0 for Adam, 0.1 per SGD/Nesterov')
    parser.add_argument('--dset', type=str, help='dataset folder: JHUIT or split0 split1 etc')
    opt = parser.parse_args()

    frozen = ''
    if opt.k == 1:
        frozen = 'not_frozen'

    if opt.k == 0:
        frozen = 'frozen'

    split = opt.dset
    folder = '/' + split + '/' + frozen + '/fc' + str(opt.nfc) + '/'

    root_weights = RESOURCES_HOME + './weights' + folder
    root_data = RESOURCES_HOME + './data_acc_loss' + folder

    if opt.model == '' and (os.path.exists(root_weights) or os.path.exists(root_data)):
        print('Directory Exists')
        exit()

    if not os.path.exists(root_weights):
        os.makedirs(root_weights)

    if not os.path.exists(root_data):
        os.makedirs(root_data)

    if __name__ == '__main__':
        Batch_size = 24

        dataset = WASHINGTON_Dataset(data_dir=RESOURCES_HOME + '/dataset/' + split + '/train_db', train=True)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=Batch_size, shuffle=True, num_workers=1)

        test_dataset = WASHINGTON_Dataset(data_dir=RESOURCES_HOME + '/dataset/' + split + '/val_db', train=False)
        testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=Batch_size, shuffle=False, num_workers=1)

    num_classes = 51
    print('classes', num_classes)

    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    # code.interact(local=locals())
    if opt.arch == 'DECO_medium_conv':
        model_0 = DECO_medium_conv(drop=opt.dropout, softmax=opt.softmax).cuda()
    elif opt.arch == 'DECO_heavy_conv':
        model_0 = DECO_heavy_conv(drop=opt.dropout, softmax=opt.softmax).cuda()
    elif opt.arch == 'DECO':
        model_0 = DECO(drop=opt.dropout, softmax=opt.softmax).cuda()
    elif opt.arch == 'DECO_senet':
        model_0 = DECO_senet(drop=opt.dropout, softmax=opt.softmax).cuda()
    else:
        print("unkown architecture!")
        exit()

    print(torch_summarize(model_0))

    acc_0 = open('%s/acc_%s_%d_%s.txt' % (root_data, opt.arch, opt.nfc, frozen), 'w+')
    loss_f_0 = open('%s/loss_%s_%d_%s.txt' % (root_data, opt.arch, opt.nfc, frozen), 'w+')
    acc_test_0 = open('%sacc_test_%s_%d_%s.txt' % (root_data, opt.arch, opt.nfc, frozen), 'w+')
    loss_f_test_0 = open('%sloss_test_%s_%d_%s.txt' % (root_data, opt.arch, opt.nfc, frozen), 'w+')

    labels_p = open('./numpy/labels_image_%s_%s.txt' % (frozen, split), 'w+')

    matrix_train = np.zeros(0)
    matrix_test = np.zeros(0)

    # Loss and Optimizer
    m = nn.LogSoftmax()
    s = nn.Softmax()
    criterion = nn.NLLLoss()
    lr = opt.lr

    if opt.k == 0:
        for param in model_0.feat.parameters():
            param.requires_grad = False

        for param in model_0.feat.fc3.parameters():
            param.requires_grad = True

    parameters_0 = itertools.ifilter(lambda p: p.requires_grad, model_0.parameters())

    if opt.solver == 'Adam':
        print("Using Adam optimizer")
        optimizer_0 = torch.optim.Adam(params=parameters_0, lr=lr, weight_decay=opt.wd)
    elif opt.solver == 'SGD':
        print("Using SGD optimizer")
        optimizer_0 = torch.optim.SGD(params=parameters_0, lr=lr, weight_decay=opt.wd, momentum=0.9)
    elif opt.solver == 'Nesterov':
        print("Using Nesterov optimizer")
        optimizer_0 = torch.optim.SGD(params=parameters_0, lr=lr, weight_decay=opt.wd, momentum=0.9, nesterov=True)
    else:
        print("wrong optimizer!")
        exit()

    savename = datetime.datetime.fromtimestamp(
        int("1284101485")
    ).strftime('%Y-%m-%d_%H:%M:%S')

    print("saving to " + savename + " file_%s_%s" % (opt.arch, frozen))

    # Training
    correct_train_0 = 0
    total_train_0 = 0

    num_epoch = opt.nepoch
    if opt.model:
        model_0.load_state_dict(torch.load(root_weights + opt.model))
        print("LOADING MODEL SNAPSHOT")

    list = ['apple', 'ball', 'banana', 'bell_pepper', 'binder', 'bowl', 'calculator', 'camera', 'cap', 'cell_phone',
            'cereal_box', 'coffee_mug', 'comb',
            'dry_battery', 'flashlight', 'food_bag', 'food_box', 'food_can', 'food_cup', 'food_jar', 'garlic',
            'glue_stick',
            'greens', 'hand_towel',
            'instant_noodles', 'keyboard', 'kleenex', 'lemon', 'lightbulb', 'lime', 'marker', 'mushroom', 'notebook',
            'onion', 'orange', 'peach', 'pear',
            'pitcher', 'plate', 'pliers', 'potato', 'rubber_eraser', 'scissors', 'shampoo', 'soda_can', 'sponge',
            'stapler',
            'tomato', 'toothbrush',
            'toothpaste', 'water_bottle']

    # training/testing loop
    for epoch in range(num_epoch):
        train(model_0, dataloader, criterion, root_weights, m, num_epoch, epoch, lr)
        if math.fmod(epoch + 1, opt.test_epochs) == 0 or epoch == num_epoch - 1:
            print("epoch: %d" % epoch)
            test(model_0, testdataloader, criterion, m, num_epoch, epoch, loss_f_test_0, s, labels_p,  acc_test_0)

            # reducing learning rate every num_epoch/3 epochs:
        if math.fmod(epoch + 1, num_epoch / 3) == 0:
            lr = lr * opt.lr_gamma  # lr_gamma usually is 0.1
            print("new learning rate: %.6f" % lr)

    # saving data
    npArray_frozen = open('./numpy/pcl_np_%s_%s' % (frozen, split), 'w+')
    np.save(npArray_frozen, matrix_test)

    acc_0.close()
    loss_f_0.close()

    acc_test_0.close()
    loss_f_test_0.close()

    labels_p.close()

    npArray_frozen.close()

if __name__ == '__main__':
    main()
