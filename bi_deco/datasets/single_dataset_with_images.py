class SingleImageDatasetWithLabels(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.dir_A = root
        self.classes, self.class_to_idx = find_classes(self.dir_A)
        self.n_classes = len(self.classes)  # number of classes
        self.data_A = make_dataset(self.dir_A, self.class_to_idx)  # in data we have both images and label

        self.files_A = sorted(glob.glob(root + '/*.*'))

    #        import pdb
    #        pdb.set_trace()

    def __getitem__(self, index):
        # getting images path and labels
        path_A, label_A = self.data_A[
            index % len(self.data_A)]  # this trick should be nice for getting an output even with index>len

        # getting actual images, packing them into a tuple with labels
        item_A = (self.transform(Image.open(path_A)), label_A)

        return item_A

    def __len__(self):
        return len(self.data_A)
