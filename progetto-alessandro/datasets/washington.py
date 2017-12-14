from __future__ import print_function
import torch.utils.data
import h5py
import os.path
import torch.utils.data as data
import torchvision as vision
from PIL import Image
from PIL import ImageChops
import numpy as np
import random


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


def load_dataset(data_dir, split, batch_size, preprocess=None):
    db_dir = os.path.join(data_dir, split)
    db_dir = os.path.join(db_dir, "train_db")
    # data_dir'+split+'/train_db'
    dataset = WASHINGTON_Dataset(data_dir=db_dir, train=True)
    if preprocess:
        dataset= preprocess(dataset)
    training_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    test_dataset = WASHINGTON_Dataset(data_dir='./dataset/'+split+'/val_db', train=False)
    if preprocess:
        test_dataset = preprocess(test_dataset)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    return training_loader, test_loader
