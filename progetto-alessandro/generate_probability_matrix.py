from __future__ import print_function

import argparse

import os

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='identification number of gpu')
parser.add_argument('--nfc', type=int, default=4096, help='number of fc2 s neurons')
parser.add_argument('--k', type=int, default=1, help='not freezed = 1')
parser.add_argument('--batchSize', type=int, default=8, help='input batch size')
parser.add_argument('--nepoch', type=int, default=60, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='cls', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dropout', type=float, default=0.0, help='percentage of neurons set to zero on FC1')
parser.add_argument('--wd', type=float, default=0.0, help='weight_decay, default to 0')
parser.add_argument('--arch', type=str, default='DECO_medium_conv',
                    help='model architecture: DECO, DECO_medium_conv, DECO_heavy_conv')
parser.add_argument('--softmax', type=bool, default=True, help='enable or disable softmax at DECO output')
parser.add_argument('--split', type=str, help='dataset folder: JHUIT or split0 split1 etc')
opt = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)
import torch.utils.data as data
import torchvision as vision
import h5py
from bi_deco.main import *
import torch.utils.data as data
from PIL import Image
from PIL import ImageChops
import os.path

frozen = 'not_freezed' if opt.k == 1 else 'freezed'
split = opt.split
folder = '/' + split + '/' + frozen + '/fc' + str(opt.nfc) + '/'

root_weights = './weights' + folder
root_np = './np' + folder

if not os.path.exists(root_np):
    os.makedirs(root_np)


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


if __name__ == '__main__':
    Batch_size = 24

    dataset = WASHINGTON_Dataset(data_dir='./dataset/' + split + '/train_db', train=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=Batch_size, shuffle=True, num_workers=1)

    test_dataset = WASHINGTON_Dataset(data_dir='./dataset/' + split + '/val_db', train=False)
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

labels_p = open('%s/labels_image_%s_%s.txt' % (root_np, frozen, split), 'w+')

# Loss and Optimizer
m = nn.LogSoftmax()
s = nn.Softmax()
criterion = nn.NLLLoss()

# num_epoch = opt.nepoch
if opt.model:
    model_0.load_state_dict(torch.load(root_weights + opt.model))
    print("LOADING MODEL SNAPSHOT")

# Test



list = ['apple', 'ball', 'banana', 'bell_pepper', 'binder', 'bowl', 'calculator', 'camera', 'cap', 'cell_phone',
        'cereal_box', 'coffee_mug', 'comb',
        'dry_battery', 'flashlight', 'food_bag', 'food_box', 'food_can', 'food_cup', 'food_jar', 'garlic', 'glue_stick',
        'greens', 'hand_towel',
        'instant_noodles', 'keyboard', 'kleenex', 'lemon', 'lightbulb', 'lime', 'marker', 'mushroom', 'notebook',
        'onion', 'orange', 'peach', 'pear',
        'pitcher', 'plate', 'pliers', 'potato', 'rubber_eraser', 'scissors', 'shampoo', 'soda_can', 'sponge', 'stapler',
        'tomato', 'toothbrush',
        'toothpaste', 'water_bottle']

matrix_test = []
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

    outputs_np = s(outputs_0)
    outputs_np = outputs_np.cpu()
    outputs_np = outputs_np.data.numpy()

    matrix_test = np.append([matrix_test], [outputs_np])

    matrix_test = matrix_test.reshape(-1, 51)

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
print("Test -->  Accuracy_0: %.6f" % (accuracy_0))

# saving data
npArray_freezed = open('%s/pcl_np_%s_%s' % (root_np, frozen, split), 'w+')
np.save(npArray_freezed, matrix_test)

# loss_f_test_0.close()

labels_p.close()

npArray_freezed.close()
