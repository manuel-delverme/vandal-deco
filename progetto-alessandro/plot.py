from __future__ import print_function
import os
import os.path
import matplotlib
matplotlib.use('Agg')
import png
from PIL import Image
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--split', type=str, default = '',  help='split0, ..., split9, JHUIT')

opt = parser.parse_args()


#Where the file are saved
root = './plot'
#Split of dataset: split0...split9,JHUIT
split = opt.split


freezed = 'freezed'
notfreezed = 'not_freezed'


#Root of file containing loss and accuracy
root_txt_freezed = './data_acc_loss/'+split+'/'+freezed+'/fc4096/'
root_txt_notfreezed = './data_acc_loss/'+split+'/'+notfreezed+'/fc4096/'

if not os.path.exists(root+'/'+split):
    os.makedirs(root+'/'+split)


acc_freezed = 'acc_DECO_medium_conv_4096_freezed.txt'
acc_freezed_test = 'acc_test_DECO_medium_conv_4096_freezed.txt'
acc_notfreezed = 'acc_DECO_medium_conv_4096_not_freezed.txt'
acc_notfreezed_test = 'acc_test_DECO_medium_conv_4096_not_freezed.txt'
loss_freezed = 'loss_DECO_medium_conv_4096_freezed.txt'
loss_freezed_test = 'loss_test_DECO_medium_conv_4096_freezed.txt' 
loss_notfreezed = 'loss_DECO_medium_conv_4096_not_freezed.txt'
loss_notfreezed_test = 'loss_test_DECO_medium_conv_4096_not_freezed.txt'


list = ['apple', 'ball', 'banana', 'bell_pepper','binder', 'bowl', 'calculator', 'camera', 'cap', 'cell_phone', 'cereal_box', 'coffee_mug', 'comb',
	'dry_battery', 'flashlight', 'food_bag', 'food_box', 'food_can', 'food_cup', 'food_jar', 'garlic', 'glue_stick', 'greens', 'hand_towel',
	'instant_noodles', 'keyboard', 'kleenex', 'lemon', 'lightbulb', 'lime', 'marker', 'mushroom', 'notebook', 'onion', 'orange', 'peach', 'pear',
	'pitcher', 'plate', 'pliers', 'potato', 'rubber_eraser', 'scissors', 'shampoo', 'soda_can', 'sponge', 'stapler', 'tomato', 'toothbrush',
	'toothpaste', 'water_bottle']



with open(root_txt_freezed+acc_freezed) as f:
 content = f.readlines()
#Instruction commented when the number of epoch of different training is equal
 #content = content[:18]
# you may also want to remove whitespace characters like `\n` at the end of each line
 for i in range(len(content)):
  if i%100 == 0:
    content = [x for x in content] 

iterations = range (len(content))


plt.plot(iterations, content) # plotting by columns
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.title('Accuracy (training)')
plt.savefig('%s/%s/acc_%s_freezed.png' %(root,split,split))
plt.close()

with open(root_txt_freezed+acc_freezed_test) as f:
 content = f.readlines()
#Instruction commented when the number of epoch of different training is equal
# content = content[:18]
# you may also want to remove whitespace characters like `\n` at the end of each line
 for i in range(len(content)):
  if i%100 == 0:
    content = [x for x in content] 
iterations = range (len(content))


plt.plot(iterations, content) # plotting by columns
plt.xlabel('Iterations')  
plt.ylabel('Accuracy')
plt.title('Accuracy (test)')
plt.savefig('%s/%s/acc_test_%s_freezed.png' %(root,split,split))
plt.close()


with open(root_txt_freezed+loss_freezed) as loss:
 content = loss.readlines()
# content = content[:81960]

 for i in range(len(content)):
  if i%100 == 0:
     content = [x for x in content]

iterations = range (len(content))

plt.plot(iterations, content)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss (training)')
plt.savefig('%s/%s/loss_%s_freezed.png' %(root,split,split))
plt.close()


with open(root_txt_freezed+loss_freezed_test) as loss:
 content = loss.readlines()
# content = content[:81960]

 for i in range(len(content)):
  if i%100 == 0:
     content = [x for x in content]

iterations = range (len(content))

plt.plot(iterations, content)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss (training)')
plt.savefig('%s/%s/loss_test_%s_freezed.png' %(root,split,split))
plt.close()


with open(root_txt_notfreezed+acc_notfreezed) as s:
 content = s.readlines()
# content = content[:81960]
# you may also want to remove whitespace characters like `\n` at the end of each line
 for i in range(len(content)):
  if i%100 == 0:
    content = [x for x in content] 

iterations = range (len(content))

plt.plot(iterations, content)
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.title('Accuracy (training)')
plt.savefig('%s/%s/acc_%s_notfreezed.png' %(root,split,split))

plt.close()

with open(root_txt_notfreezed+acc_notfreezed_test) as s:
 content = s.readlines()
# content = content[:81960]
# you may also want to remove whitespace characters like `\n` at the end of each line
 for i in range(len(content)):
  if i%100 == 0:
    content = [x for x in content] 

iterations = range (len(content))

plt.plot(iterations, content)
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.title('Accuracy (test)')
plt.savefig('%s/%s/acc_test_%s_notfreezed.png' %(root,split,split))

plt.close()


with open(root_txt_notfreezed+loss_notfreezed) as lossnf:
 content = lossnf.readlines()
# content = content[:81960]
 for i in range(len(content)):
  if i%100 == 0:
     content = [x for x in content]

iterations = range(len(content))
plt.plot(iterations, content)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss (training)')
plt.savefig('%s/%s/loss_%s_notfreezed.png' %(root,split,split))
plt.close()

with open(root_txt_notfreezed+loss_notfreezed_test) as lossnf:
 content = lossnf.readlines()
# content = content[:81960]
 for i in range(len(content)):
  if i%100 == 0:
     content = [x for x in content]

iterations = range(len(content))
plt.plot(iterations, content)
plt.xlabel('Iterations')  
plt.ylabel('Loss')
plt.title('Loss (test)')
plt.savefig('%s/%s/loss_test_%s_notfreezed.png' %(root,split,split))
plt.close()

