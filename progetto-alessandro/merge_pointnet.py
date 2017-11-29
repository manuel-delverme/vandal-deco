import pickle
from argparse import ArgumentParser
import numpy as np

"""python merge_pointnet.py tmp/washington_deco_s0_depth.pkl pcl_np_freezed_split0 
/home/paolo/DepthNet/scripts/Washington/splits/depth_test_split_0.txt /home/paolo/DepthNet/scripts/Washington/splits/depth_test_split_0.txt 51"""


score2 = 0

def load_split(split_path, feat_path, classes):
    ft_lines = open(split_path, 'rt').readlines()
    feat_dict = pickle.load(open(feat_path))
    features = []
    labels = []

    for line in ft_lines:
        [path, classLabel] = line.split()
        nClass = int(classLabel)
        labels.append(nClass)
        features.append(feat_dict[path].values()[0].ravel())
    labels = np.vstack(labels).reshape(1,-1)
    features = np.vstack(features)
    return (features, labels)


def get_args():
    parser = ArgumentParser()
    parser.add_argument("rgb_feat")
    parser.add_argument("depth_feat")
    parser.add_argument("split_file")
    parser.add_argument("split_file_depth")
    parser.add_argument("n_classes", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    p1, true_labels = load_split(args.split_file, args.rgb_feat, args.n_classes)
    p2 = np.load(args.depth_feat)
    p2 = p2[:,:49]
    print(true_labels)
    # import ipdb; ipdb.set_trace()
    score1 = np.sum(true_labels == np.argmax(p1, axis=1)) / float(true_labels.size)
    score2 = np.sum(true_labels == np.argmax(p2, axis=1)) / float(true_labels.size)
    print("RGB: %.2f; Depth: %.2f\n" % (score1*100, score2*100))
    res = []
    max_val = 0
    for x in np.arange(0, 1, 0.05):
        s = np.sum(true_labels == np.argmax(p1*x + p2*(1-x), axis=1)) / float(true_labels.size)
        if s > max_val:
            max_val = s
        res.append((x, s))
    #for (x, score) in res:
    #    print("Coeff: %f - score: %.2f" % (x, score*100))
    print max_val
