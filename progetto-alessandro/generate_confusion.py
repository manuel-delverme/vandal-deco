from __future__ import print_function
import pickle
import tqdm
import sklearn.metrics
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
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# RESOURCES_HOME = "/home/iodice/vandal-deco/progetto-alessandro/tesi/tesi/"
RESOURCES_HOME = "/home/alessandrodm/tesi/"
RESULTS_HOME = "/home/iodice/alessandro_results/"


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


def test(model, dataset_loader):
    model.eval()
    y_hat = []
    y_gt = []

    for i, (images, labels) in tqdm.tqdm(enumerate(dataset_loader), total=len(dataset_loader)):
        images = Variable(images, volatile=True)
        labels = Variable(labels, volatile=True)
        labels = labels.long()
        outputs_0, _ = model(images)

        predicted_0 = outputs_0.data.max(1)[1]
        y_hat.extend(predicted_0)
        y_gt.extend(labels.data)

    return y_hat, y_gt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default="-1", help='identification number of gpu, -1 for cpu')
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
                        help='model solver: Adam, SGD or Nesterov. For nesterov is suggested: generate_confusion.py --nepoch 120 --solver Nesterov --lr 0.01  --lr_gamma 0.1')
    parser.add_argument('--lr', type=float, default=0.001, help='starting learning rate.')
    parser.add_argument('--lr_gamma', type=float, default=1.0,
                        help='learning rate decrease factor.Suggested 1.0 for Adam, 0.1 per SGD/Nesterov')
    parser.add_argument('--dset', type=str, help='dataset folder: JHUIT or split0 split1 etc')
    opt = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "" if opt.gpu == "-1" else opt.gpu

    frozen = ''
    if opt.k == 1:
        frozen = 'not_frozen'

    if opt.k == 0:
        frozen = 'frozen'

    split = opt.dset
    folder = '/' + split + '/' + frozen + '/fc' + str(opt.nfc) + '/'

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

    test_dataset = WASHINGTON_Dataset(data_dir=RESOURCES_HOME + '/dataset/' + split + '/val_db', train=False)
    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=Batch_size, shuffle=False, num_workers=1)

    num_classes = 51
    print('classes', num_classes)

    try:
        os.makedirs(opt.outf)
    except OSError as e:
        pass

    if opt.arch == 'DECO_medium_conv':
        model_0 = DECO_medium_conv(drop=opt.dropout, softmax=opt.softmax)
    elif opt.arch == 'DECO_heavy_conv':
        model_0 = DECO_heavy_conv(drop=opt.dropout, softmax=opt.softmax)
    elif opt.arch == 'DECO':
        model_0 = DECO(drop=opt.dropout, softmax=opt.softmax)
    elif opt.arch == 'DECO_senet':
        model_0 = DECO_senet(drop=opt.dropout, softmax=opt.softmax)
    else:
        print("unkown architecture!")
        raise NotImplementedError()

    print(torch_summarize(model_0))

    # Loss and Optimizer
    if opt.k == 0:
        for param in model_0.feat.parameters():
            param.requires_grad = False

        for param in model_0.feat.fc3.parameters():
            param.requires_grad = True

    model_0.load_state_dict(torch.load(root_weights + opt.model, map_location=lambda storage, loc: storage))
    y_hat, y_gt = test(model_0, testdataloader)
    with open("confusion_data.pkl", "wb") as fout:
        pickle.dump((y_gt, y_hat), fout)

    # cnf_matrix = sklearn.metrics.confusion_matrix(y_gt, y_hat)
    # np.set_printoptions(precision=2)
    # # Plot non-normalized confusion matrix
    # plt.figure(figsize=(1000, 1000))
    # plot_confusion_matrix(cnf_matrix, classes=label_names, title='Confusion matrix, without normalization')
    # plt.savefig('unnormalized_confusion_matrix.png')


if __name__ == '__main__':
    main()
