from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os, sys, shutil, time, random
import argparse
import torch
import torch.backends.cudnn as cudnn
import math

import numpy as np
import os
import copy
import torch
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import os
import time
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import torch.backends.cudnn as cudnn
from nni.algorithms.compression.v2.pytorch.pruning import FPGMPruner, L1NormPruner
from tqdm import tqdm
import subprocess
from prepare_wider_data import wider_data_file
from data.config import cfg
from EXTD_64 import build_extd
from layers.modules.multibox_loss import MultiBoxLoss
from data.factory import dataset_factory, detection_collate
import wider_test
import eval_tools.evaluation as evaluation
import numpy as np
from argparse import Namespace
import wandb
import pickle
import yaml
from bayesian_pruner import MultiLayerPruner
from utilities import *

# compile
os.system("python3 ./eval_tools/bbox_setup.py build_ext --inplace")
print('compile completed')

parser = argparse.ArgumentParser(description='Stop pruning and fine tune the model without changing pruned weights',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Optimization options
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size.')
parser.add_argument('--num_workers', default=0, type=int, help='Number of workers used in dataloading')
parser.add_argument('--learning_rate', type=float, default=0.001, help='The Learning Rate.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', type=float, default=0.0005, help='Weight decay (L2 penalty).')
parser.add_argument('--schedule', type=int, nargs='+', default=[10],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gammas', type=float, nargs='+', default=[0.1],
                    help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')
parser.add_argument('--pretrained_model', type=str,
                    default='',
                    help='path to the pretrained model')
parser.add_argument('--pruning_rate', type=float, default=0.2, help='sparsity per layer')

args = parser.parse_args()
cudnn.benchmark = True
use_cuda = True


def main():
    """ Prune and finetune the model iteratively """
    seed = 0
    set_all_seeds(seed)

    # Init dataset
    print('prepare wider')
    wider_data_file()

    # Datasets and dataloaders
    train_dataset, val_dataset = dataset_factory('face')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, collate_fn=detection_collate, pin_memory=False,
                                               generator=torch.Generator(device='cuda'))
    val_loader = data.DataLoader(val_dataset, args.batch_size, num_workers=args.num_workers, shuffle=False,
                                 collate_fn=detection_collate, pin_memory=False)

    # Init model, criterion, and optimizer
    net = build_extd('train', cfg.NUM_CLASSES)
    print('Load network....')
    net.load_state_dict(torch.load(args.pretrained_model))
    print('Network loaded successfully')

    # define loss function (criterion) and optimizer
    criterion = MultiBoxLoss(cfg, 'face', use_cuda)
    optimizer = torch.optim.SGD(net.parameters(), lr=args.learning_rate, momentum=args.momentum,
                                weight_decay=args.decay)
    net.cuda()
    criterion.cuda()

    # Wandb logging
    wandb_config = {'epochs': args.epochs, 'lr_steps': args.schedule, 'batch_size': args.batch_size, 'pruner': 'FPGM',
                    'Pre-trained model': args.pretrained_model, 'pruning_rate': args.pruning_rate,
                    'n_accumulation_step': 1, 'momentum': args.momentum, 'gammas': args.gammas,
                    'lr': args.learning_rate, 'weight_decay': args.decay, 'seed': seed}
    run_name = f'{time.strftime("%b_%d_%Y_%H_%M_%S", time.localtime())}'
    save_dir = f'logs/Stop/{run_name}'
    os.makedirs(save_dir, exist_ok=False)
    run = wandb.init(project='', entity="", name=run_name, config=wandb_config)

    # Prune
    # Load Bayesian optimization results
    bayesian_pruning_log_dir = ''
    with open(f'{bayesian_pruning_log_dir}/optimal_pruning_rates.pkl', 'rb') as f:
        optimal_pruning_rates = pickle.load(f)
    with open(f'{bayesian_pruning_log_dir}/pruning_groups.yaml', 'r') as f:
        pruning_groups = yaml.safe_load(f)
    pruning_dict = {tuple(k): v for k, v in zip(pruning_groups, optimal_pruning_rates)}
    pruner = MultiLayerPruner(pruning_dict)
    pruner.prune(net)
    print('Model pruned')

    # Main loop
    for epoch in tqdm(range(args.epochs)):
        current_learning_rate = step(optimizer, epoch, args.gammas, args.schedule)
        train(train_loader, net, criterion, optimizer, epoch, current_learning_rate, wandb_config)
        torch.save(net.state_dict(), f'{save_dir}/EXTD-Stop-{epoch}.pth')

        # Validation
        sparsity = calc_sparsity(net)
        validation(net, criterion, optimizer, epoch, wandb_config, val_loader, sparsity, save_model=False)


def train(train_loader, model, criterion, optimizer, epoch, current_lr, config):
    model.train()
    epoch_running_loss1, epoch_running_loss2 = 0.0, 0.0
    for batch_idx, (images, targets) in enumerate(train_loader):
        images = images.cuda()
        targets = [ann.cuda() for ann in targets]

        # Loss
        out = model(images)
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c  # stress more on loss_l
        loss = loss / config['n_accumulation_step']
        loss.backward()

        if ((batch_idx + 1) % config['n_accumulation_step'] == 0) or (batch_idx + 1 == len(train_loader)):
            optimizer.step()
            optimizer.zero_grad()

        epoch_running_loss1 += loss_l.item() / config['n_accumulation_step']
        epoch_running_loss2 += loss_c.item() / config['n_accumulation_step']

    # Logging
    epoch_running_loss1 /= len(train_loader)
    epoch_running_loss2 /= len(train_loader)
    wandb.log({'Epoch': epoch, 'Train/Epoch_loss1': epoch_running_loss1, 'Train/Epoch_loss2': epoch_running_loss2,
               'Train/Loss': epoch_running_loss1 + epoch_running_loss2, 'Learning_rate': current_lr})


def step(optimizer, epoch, gammas, schedule):
    # Sets the learning rate to the initial LR decayed by 10 every 30 epochs
    lr = args.learning_rate
    assert len(gammas) == len(schedule), "length of gammas and schedule should be equal"
    for (gamma, step) in zip(gammas, schedule):
        if (epoch >= step):
            lr = lr * gamma
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__ == '__main__':
    main()
