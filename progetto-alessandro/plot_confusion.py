from __future__ import print_function

import itertools
import pickle

# matplotlib.use('GTK')
import matplotlib.pyplot as plt
import sklearn.metrics

from bi_deco.main import *


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if cm[i, j] > 0:
            plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')




def main():
    with open("confusion_data.pkl", "rb") as fin:
        y_gt, y_hat = pickle.load(fin)
    cnf_matrix = sklearn.metrics.confusion_matrix(y_gt, y_hat)
    np.set_printoptions(precision=2)
    label_names = [
        'apple', 'ball', 'banana', 'bell_pepper', 'binder', 'bowl', 'calculator', 'camera', 'cap', 'cell_phone',
        'cereal_box', 'coffee_mug', 'comb', 'dry_battery', 'flashlight', 'food_bag', 'food_box', 'food_can',
        'food_cup', 'food_jar', 'garlic', 'glue_stick', 'greens', 'hand_towel', 'instant_noodles', 'keyboard',
        'kleenex', 'lemon', 'lightbulb', 'lime', 'marker', 'mushroom', 'notebook', 'onion', 'orange', 'peach',
        'pear', 'pitcher', 'plate', 'pliers', 'potato', 'rubber_eraser', 'scissors', 'shampoo', 'soda_can',
        'sponge', 'stapler', 'tomato', 'toothbrush', 'toothpaste', 'water_bottle'
    ]
    # plt.figure(figsize=(1000, 1000))
    import ipdb; ipdb.set_trace()
    np.fill_diagonal(cnf_matrix, 0)
    plot_confusion_matrix(cnf_matrix, classes=label_names, title='Confusion matrix, without normalization')
    plt.show()
    # plt.savefig('unnormalized_confusion_matrix.png')


if __name__ == '__main__':
    main()
