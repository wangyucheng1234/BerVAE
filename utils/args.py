import os
import time

dataset = 'fcv'  # type=str, 'yfcc | fcv |
workers = 0 # type=int, number of data loading workers, default=0
batch_size = 256 #type=int, default=16, help='input batch size'
num_epochs = 200 # type=int, default=50, help='number of epochs to train for'
use_cuda = True
use_checkpoint = False
lr = 3e-4 #type=float, default=0.0001
lr_decay_rate = 20 #type=float, default=30
single_lr_decay_rate = 20 #type=float, default=30
weight_decay = 1e-4 #type=float, default=1e-4
nbits = 64
feature_size = 4096 #Length of feature representation for each video frame
max_frames = 25
hidden_size = 256
test_batch_size = 256
nnachors = 2000

prior_prob = 0.5
pretrain_epoch = 5
annealing_rate = 0.1
eps = 1e-8

data_root = './fcv/'  #Todo: set your path to the dataset here
home_root = './BerVAE/' #Todo: set your path to your home menu
train_feat_path = data_root+'fcv_train_feats.h5'
test_feat_path = data_root+'fcv_test_feats.h5'
label_path = data_root+'fcv_test_labels.mat'
save_dir = home_root+'models/' + dataset
file_path = save_dir + '_bits_' + str(nbits)
