import argparse
import os
import random

import numpy as np
import torch

from cimfe.engine.exp_scrssl import Exp_SCRSSL

os.environ['CUDA_LAUNCH_BLOCKING'] = '0'


parser = argparse.ArgumentParser(description='SCRSSL self-supervised pretraining')

# basic config
parser.add_argument('--seed', type=int, default=2025, help='random seed')

# data loader
parser.add_argument('--data_root', type=str, default='./data', help='root directory containing label.xlsx and preprocessed_1m/')
parser.add_argument('--pretrain_checkpoints', type=str, default='./outputs/pretrain_checkpoints/', help='where to save pretraining checkpoints')
parser.add_argument('--load_checkpoints', type=str, default=None, help='optional checkpoint to resume pretraining from')
parser.add_argument('--train_ratio', type=float, default=1, help='fraction of sliding windows used for training (remainder is validation)')

# forecasting task
parser.add_argument('--seq_len', type=int, default=60, help='input sequence length')

# model define
parser.add_argument('--d_model', type=int, default=64, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=4, help='num of heads')
parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
parser.add_argument('--d_ff', type=int, default=256, help='dimension of fcn')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--head_dropout', type=float, default=0.1, help='head dropout')
parser.add_argument('--patch_len', type=int, default=3, help='patch length')
parser.add_argument('--stride', type=int, default=3, help='stride')

parser.add_argument('--pre_norm', default=False, action='store_true', help='pre layer norm')
parser.add_argument('--norm', default='BatchNorm', type=str, help='normalization layer')

# optimization
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--pretrain_epochs', type=int, default=100, help='train epochs')
parser.add_argument('--pretrain_learning_rate', type=float, default=0.0005, help='optimizer learning rate')

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')

parser.add_argument('--mask_ratio', type=float, default=0.4)
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--pretrain_task', type=str, default='rec', choices=['rec', 'psi'])

args = parser.parse_args()
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
args.num_patch = args.seq_len // args.patch_len
args.learning_rate = args.pretrain_learning_rate

print('Args in experiment:')
print(args)
print(torch.cuda.device_count())

fix_seed = args.seed
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

for ii in range(args.itr):
    # setting record of experiments
    pretrain_setting = '{}_dm{}_df{}_nh{}_el{}_pl{}_pre{}_{}_ep{}_plr{}_alp{}'.format(
        args.pretrain_task,
        args.d_model,
        args.d_ff,
        args.n_heads,
        args.e_layers,
        args.patch_len,
        args.pre_norm,
        args.norm,
        args.pretrain_epochs,
        args.pretrain_learning_rate,
        args.alpha,
    )

    exp = Exp_SCRSSL(args)
    print('>>>>>>>start pre_training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(pretrain_setting))
    exp.pretrain(pretrain_setting)
    torch.cuda.empty_cache()
