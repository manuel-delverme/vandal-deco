import glob
import bi_deco
import scipy.ndimage as nd
import math
import torchvision as vision
import torch.utils.data
import h5py
from PIL import Image
import os.path
import collections


def get_trainable_params(model):
    params = []
    trainable = collections.defaultdict(int)
    for name, p in model.named_parameters():
        if p.requires_grad:
            network_name = name.split(".")[0]
            trainable[network_name] += 1
            params.append(p)
    print("trainable parameters:",)
    for k,v in trainable.items():
        print(k, v)
    return params


def show_memusage(device=2):
    import gpustat
    gpu_stats = gpustat.GPUStatCollection.new_query()
    item = gpu_stats.jsonify()["gpus"][device]
    print("GPU: {}/{}".format(item["memory.used"], item["memory.total"]))


def latest_checkpoint_path(experiment_name):
    experiment_epoch = -1
    for checkpoint_path in glob.glob("/home/iodice/vandal-deco/state_dicts/{}cls_model_*.pth".format(experiment_name)):
        f, t = checkpoint_path.rfind("_") + 1, -4
        assert (f > 0)
        checkpoint_epoch = checkpoint_path[f:t]
        experiment_epoch = max(int(checkpoint_epoch), experiment_epoch)
    if experiment_epoch == -1:
        raise Exception("checkpoint {} not found".format(experiment_name))
    checkpoint_path = "/home/iodice/vandal-deco/state_dicts/{}cls_model_{}.pth".format(experiment_name,
                                                                                       experiment_epoch)
    return checkpoint_path, experiment_epoch


class Dataset_depth(torch.utils.data.Dataset):
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


def load_dataset(split, batch_size):
    dataset = Dataset_depth(data_dir='/scratch/dataset/split{}/train_db/'.format(split), train=True)
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        # pin_memory=True
    )
    test_dataset = Dataset_depth(data_dir='/scratch/dataset/split{}/val_db/'.format(split), train=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True,
        # pin_memory=True,
    )

    # train_loader, test_loader = bi_deco.datasets.washington.load_dataset(
    #     data_dir='/scratch/dataset/',
    #     split=split,
    #     batch_size=batch_size,
    # )
    return test_loader, train_loader
