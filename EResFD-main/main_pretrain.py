"""
B-FPGM
Pre-training of the EResFD model
CERTH 2024
"""

# -*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import time
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from nni.algorithms.compression.v2.pytorch.pruning import FPGMPruner

import tensorflow as tf

from prepare_wider_data import wider_data_file

from data.config import cfg
from models.eresfd import build_model
from layers.modules.multibox_loss import MultiBoxLoss
from data.factory import dataset_factory, detection_collate
from test_wider import predict_wider
from tqdm import tqdm
from utilities import *

# compile
os.system("python3 bbox_setup.py build_ext --inplace")
# print('compile completed')

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.autograd.set_detect_anomaly(True)

print('argparse')
parser = argparse.ArgumentParser(
    description='EResFD face Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='face', choices=['hand', 'face', 'head'], help='Train target')
parser.add_argument('--batch_size', default=16, type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=0, type=int, help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--multigpu', default=False, type=str2bool, help='Use mutil Gpu training')
parser.add_argument('--eval_verbose', default=True, type=str2bool, help='Use mutil Gpu training')
args = parser.parse_args()

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


# gflops, mflops = compute_flops(net, np.array([cfg.INPUT_SIZE, cfg.INPUT_SIZE]))
# print('# of params in Classification model: %d, flops: %.2f GFLOPS, %.2f MFLOPS, image_size: %d' % \
#       (sum([p.data.nelement() for p in net.parameters()]), gflops, mflops, cfg.INPUT_SIZE))


def train():
    """ Trains and validates the network """
    seed = 0
    set_all_seeds(seed)

    # dataset setting
    print('prepare wider')
    wider_data_file()

    # Datasets and dataloaders
    train_dataset, val_dataset = dataset_factory('face')
    train_loader = data.DataLoader(train_dataset, args.batch_size, num_workers=args.num_workers, shuffle=True,
                                   collate_fn=detection_collate, pin_memory=False,
                                   generator=torch.Generator(device='cuda'))
    val_loader = data.DataLoader(val_dataset, args.batch_size, num_workers=args.num_workers, shuffle=False,
                                 collate_fn=detection_collate, pin_memory=False)

    # Network
    net = build_model("train", cfg.NUM_CLASSES, width_mult=0.0625)
    print('Initialize base network....')
    net.base.apply(net.weights_init)
    print('Initializing weights...')
    net.loc.apply(net.weights_init)
    net.conf.apply(net.weights_init)
    net = net.cuda()
    cudnn.benckmark = True

    # Optimizer
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = MultiBoxLoss(cfg, args.dataset, args.cuda)

    # Wandb logging
    wandb_config = {'epochs': cfg['EPOCHES'], 'lr_steps': cfg['LR_STEPS'], 'batch_size': args.batch_size,
                    'n_accumulation_step': 1, 'gamma': args.gamma, 'lr': args.lr, 'weight_decay': args.weight_decay,
                    'seed': seed}
    run_name = f'{time.strftime("%b_%d_%Y_%H_%M_%S", time.localtime())}'
    save_dir = f'logs/Train/{run_name}'
    os.makedirs(save_dir, exist_ok=False)
    run = wandb.init(project='PROJECT', entity="ENTITY", name=run_name, config=wandb_config)

    step_index, iteration = 0, 0
    min_loss = np.inf
    current_lr = args.lr

    for epoch in tqdm(range(cfg.EPOCHES)):
        net.train()
        epoch_running_loss1, epoch_running_loss2 = 0.0, 0.0
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.cuda()
            targets = [ann.cuda() for ann in targets]

            if iteration in cfg.LR_STEPS:
                step_index += 1
                current_lr = adjust_learning_rate(optimizer, args.gamma, step_index)

            # Loss
            loss_l, loss_c = wrapper(images, net, criterion, targets)
            loss = loss_l + loss_c
            loss = loss / wandb_config['n_accumulation_step']
            loss.backward()

            if ((batch_idx + 1) % wandb_config['n_accumulation_step'] == 0) or (batch_idx + 1 == len(train_loader)):
                optimizer.step()
                optimizer.zero_grad()

            epoch_running_loss1 += loss_l.item() / wandb_config['n_accumulation_step']
            epoch_running_loss2 += loss_c.item() / wandb_config['n_accumulation_step']

            iteration += 1
        # Logging
        epoch_running_loss1 /= len(train_loader)
        epoch_running_loss2 /= len(train_loader)
        wandb.log({'Epoch': epoch, 'Train/Epoch_loss1': epoch_running_loss1, 'Train/Epoch_loss2': epoch_running_loss2,
                   'Train/Loss': epoch_running_loss1 + epoch_running_loss2, 'Learning_rate': current_lr})

        # Validation
        min_loss = validation(net, criterion, optimizer, epoch, wandb_config, val_loader, min_loss=min_loss,
                              save_dir=save_dir)
        # if iteration == cfg.MAX_STEPS:
        #     break


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__ == '__main__':
    start = time.perf_counter()
    train()

    wandb.finish()
    print(f"\nExecution time = {time.perf_counter() - start:.2f} second(s)")
