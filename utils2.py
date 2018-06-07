import lmdb
import scipy.misc
from scipy import ndimage
# import caffe
import numpy as np
from StringIO import StringIO
import PIL.Image
from PIL import Image


class ReadLmdb2():
    def __init__(self, lmdb_dir):
        self.cursor = lmdb.open(lmdb_dir, readonly=True).begin().cursor()
        self.datum = caffe.proto.caffe_pb2.Datum()
        self.l=[] #the list that will keep all key values, so that I can get the corresponding key with l[index]
        for key, _ in self.cursor:
            self.l.append(key)
        self.length = len(self.l)
    def __getitem__(self,index):
        self.cursor.set_key(self.l[index])
        value = self.cursor.value()
        self.datum.ParseFromString(value)
        s = StringIO()
        s.write(self.datum.data)
        s.seek(0)
        return PIL.Image.open(s), self.datum.label


def read_lmdb_index(lmdb_dir,index):
    cursor = lmdb.open(lmdb_dir, readonly=True).begin().cursor()
    datum = caffe.proto.caffe_pb2.Datum()
    l=[] #the list that will keep all key values, so that I can get the corresponding key with l[index]
    for key, _ in cursor:
        l.append(key)
    while True:
        cursor.set_key(l[index])
        value = cursor.value()
        datum.ParseFromString(value)
        s = StringIO()
        s.write(datum.data)
        s.seek(0)
        yield np.array(PIL.Image.open(s)), datum.label


def load_lmdb_datasets(image_dim_ordering, lmdb_dir, size=64,n_channels=1, max=100000):
    X=np.zeros((max,size,size,n_channels)) #tf dim order becaouse scipy imresize works with this
    Y=np.zeros((max,1))
    i=0
    for im, label in read_lmdb(lmdb_dir):
        im=scipy.misc.imresize(im, (size, size))
        if n_channels == 3: #it already has dimension (size,size,3)
            X[i]=im
        else:
            X[i]=np.expand_dims(im,4) #adding axis, so (size,size) --> (size,size,1)
        Y[i]=label  
        i=i+1
        if i==max:
            break
    X = X.astype('float32')
    if image_dim_ordering == 'th':
        X=np.moveaxis(X, -1, 1) #switching from current tensorflow dim ordering to theano dim ordering
    #X = normalization(X, image_dim_ordering)
    nb_classes = len(np.unique(Y))
    Y_ = np_utils.to_categorical(Y, nb_classes)

    #if n_channels == 1:# TURNING 1 CHANNEL DATASET INTO 3-CHANNELS
    #    np.concatenate([X,X,X],axis=1)
    return X, Y_, nb_classes

def getIndex(list_index):
    if len(list_index) == 1:
        return list_index.pop()
    n = np.random.randint(0, len(list_index)-1)
    index = list_index.pop(n)
    return index
