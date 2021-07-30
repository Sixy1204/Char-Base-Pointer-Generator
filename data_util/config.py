# config 
import os
import torch
from numpy import random


root_dir = "../data/original_data/original_subset"
train_data_path = os.path.join(root_dir, "train_sub.npy")
#eval_data_path = os.path.join(root_dir, "val.bin")
decode_data_path = os.path.join(root_dir, "test_sub.npy")
vocab_path = os.path.join(root_dir, "vocab")
log_root = "../logs/"
if not os.path.isdir(log_root):
    os.makedirs(log_root)

is_ner = False

# Hyperparameters
hidden_dim= 256
emb_dim= 128
batch_size= 32 #mini batch
max_enc_steps= 400
max_dec_steps= 50
beam_size= 4
min_dec_steps= 5
vocab_size= 50000

#lr = 1e-3
lr = 0.15
adagrad_init_acc= 0.1

adam_eps = 1e-8
grad_acc_step = 1

rand_unif_init_mag= 0.02
trunc_norm_init_std= 1e-4
max_grad_norm= 2.0

pointer_gen = True
is_coverage = False
cov_loss_wt = 1.0

eps = 1e-12
max_iterations = 350000
use_gpu=True

lr_coverage = 0.15
#lr_coverage = 1e-3

SEED = 1234
random.seed(SEED)
