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
import scipy.ndimage as nd


class Dataset_depth(data.Dataset):
    def __init__(self, data_dir, image_size=256, train=True):
        self.image_size = image_size
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

        IMAGE_SIZE = 228
        self.rc = vision.transforms.RandomCrop(
            [IMAGE_SIZE, IMAGE_SIZE])  # crea delle varibili per ogni metodo per richiamarle piu facilmente
        self.cc = vision.transforms.CenterCrop(
            [IMAGE_SIZE, IMAGE_SIZE])  # randomcop taglia l immagine casualmente nelle dimensioni che gli ho passato
        self.resize = vision.transforms.Scale([IMAGE_SIZE, IMAGE_SIZE])
        self.toTensor = vision.transforms.ToTensor()
        self.toPIL = vision.transforms.ToPILImage()
        self.flip = vision.transforms.RandomHorizontalFlip()

    def __getitem__(self, index):  # restituisce l immagine in posizione index che gli passo come parametro e lo passa modificato
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


def load_dataset(data_dir, split, batch_size, preprocess=None, rgb=False):
    db_dir = os.path.join(data_dir, "split" + str(split))
    train_db_dir = os.path.join(db_dir, "train_db")
    test_db_dir = os.path.join(db_dir, "val_db")

    # dataset = WASHINGTON_Dataset(data_dir=train_db_dir, train=True, rgb=rgb)
    dataset = Dataset_depth(data_dir=train_db_dir, train=True)

    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=Batch_size, shuffle=True, num_workers=1)
    training_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
    )

    # test_dataset = WASHINGTON_Dataset(data_dir=test_db_dir, train=False)
    test_dataset = Dataset_depth(data_dir=test_db_dir, train=False)

    # testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=Batch_size, shuffle=False, num_workers=1)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
    )
    return training_loader, test_loader
