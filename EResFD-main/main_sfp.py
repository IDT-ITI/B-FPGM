"""
B-FPGM
Iteratively soft-prune and train a pre-trained models
CERTH 2024
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import time
import torch
import argparse
import torch.optim as optim
import torch.utils.data as data
from nni.algorithms.compression.v2.pytorch.pruning import FPGMPruner, L1NormPruner
from prepare_wider_data import wider_data_file
from data.config import cfg
from models.eresfd import build_model
from layers.modules.multibox_loss import MultiBoxLoss
from data.factory import dataset_factory, detection_collate
from test_wider import predict_wider
from os import makedirs
from tqdm import tqdm
from bayesian_pruner import *
from utilities import *
from bayesian_pruner import MultiLayerPruner


# compile
os.system("python3 bbox_setup.py build_ext --inplace")
print('compile completed')

parser = argparse.ArgumentParser(description='Training with Pruning',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size.')
parser.add_argument('--num_workers', default=0, type=int, help='Number of workers used in dataloading')
parser.add_argument('--learning_rate', type=float, default=0.001, help='The Learning Rate.')
parser.add_argument('--decay', type=float, default=0.0005, help='Weight decay (L2 penalty).')
parser.add_argument('--schedule', type=int, nargs='+', default=[100, 150],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gammas', type=float, nargs='+', default=[0.1, 0.1],
                    help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')
parser.add_argument('--pretrained_model', type=str,
                    default='',
                    help='path to the pretrained model')
parser.add_argument('--pruning_rate', type=float, default=0.2, help='pruning rate')
parser.add_argument('--epoch_prune', type=int, default=5, help='compress layer of model')

args = parser.parse_args()


def main():
    """ Trains and soft-prunes the model iteratively """
    seed = 0
    set_all_seeds(seed)

    # Init dataset
    print('prepare wider')
    wider_data_file()

    # Datasets and dataloaders
    train_dataset, val_dataset = dataset_factory('face')
    train_loader = data.DataLoader(train_dataset, args.batch_size, num_workers=args.num_workers, shuffle=True,
                                   collate_fn=detection_collate, pin_memory=False,
                                   generator=torch.Generator(device='cuda'))
    val_loader = data.DataLoader(val_dataset, args.batch_size, num_workers=args.num_workers, shuffle=False,
                                 collate_fn=detection_collate, pin_memory=False)

    # Init model, criterion, and optimizer
    net = build_model('train', cfg.NUM_CLASSES, width_mult=0.0625)
    print('Load network....')
    net.load_state_dict(torch.load(args.pretrained_model))  # our reproduced pre-trained model
    print('Network loaded successfully')

    # define loss function (criterion) and optimizer
    criterion = MultiBoxLoss(cfg, 'face', True)
    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.decay)
    net.cuda()

    config_list = [{
        'sparsity_per_layer': args.pruning_rate,
        'op_types': ['Conv2d'],
    }, {
        'exclude': True,
        'op_names': [
            'loc.0', 'loc.1', 'loc.2', 'loc.3', 'loc.4', 'loc.5',
            'conf.0', 'conf.1', 'conf.2', 'conf.3', 'conf.4', 'conf.5'
        ]
    }]

    # Wandb logging
    wandb_config = {'epochs': args.epochs, 'lr_steps': args.schedule, 'batch_size': args.batch_size, 'pruner': 'FPGM',
                    'use_bayesian': True, 'bayesian_pruning_log_dir': '', 'Pre-trained model': args.pretrained_model,
                    'pruning_rate': args.pruning_rate, 'n_accumulation_step': 1, 'epoch_prune': args.epoch_prune,
                    'gammas': args.gammas, 'lr': args.learning_rate, 'weight_decay': args.decay, 'seed': seed}
    run_name = f'{time.strftime("%b_%d_%Y_%H_%M_%S", time.localtime())}'
    save_dir = f'logs/SFP/{run_name}'
    makedirs(save_dir, exist_ok=False)
    run = wandb.init(project='PROJECT', entity="ENTITY", name=run_name, config=wandb_config)

    # Main loop
    for epoch in tqdm(range(args.epochs)):
        net.train()
        current_learning_rate = step(optimizer, epoch, args.gammas, args.schedule)

        model_path = f'{save_dir}/ERes-sfp-{epoch - 1}.pth'
        if os.path.exists(model_path):
            net = build_model('train', cfg.NUM_CLASSES, width_mult=0.0625)
            net.load_state_dict(torch.load(model_path))
            optimizer = optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.decay)

        losses = 0
        train(train_loader, net, criterion, optimizer, epoch, losses, current_learning_rate, wandb_config)

        if epoch % args.epoch_prune == 0 or epoch == args.epochs - 1:
            if wandb_config['use_bayesian']:
                with open(f'{wandb_config["bayesian_pruning_log_dir"]}/optimal_pruning_rates.pkl', 'rb') as f:
                    optimal_pruning_rates = pickle.load(f)
                with open(f'{wandb_config["bayesian_pruning_log_dir"]}/pruning_groups.yaml', 'r') as f:
                    pruning_groups = yaml.safe_load(f)
                pruning_dict = {tuple(k): v for k, v in zip(pruning_groups, optimal_pruning_rates)}
                pruner = MultiLayerPruner(pruning_dict, pruner=wandb_config['pruner'])
                pruner.prune(net)
            else:
                if wandb_config['pruner'] == 'l1':
                    pruner = L1NormPruner(net, config_list)
                elif wandb_config['pruner'] == 'FPGM':
                    pruner = FPGMPruner(net, config_list)
                else:
                    raise ValueError('Pruner not supported')
                pruner.compress()
                pruner._unwrap_model()
            print('Model pruned')

        torch.save(net.state_dict(), f'{save_dir}/ERes-sfp-{epoch}.pth')

        sparsity = calc_sparsity(net)
        validation(net, criterion, optimizer, epoch, wandb_config, val_loader, sparsity,
                   save_model=False)


def train(train_loader, model, criterion, optimizer, epoch, losses, current_lr, config):
    model.train()
    epoch_running_loss1, epoch_running_loss2 = 0.0, 0.0
    for batch_idx, (images, targets) in enumerate(train_loader):
        images = images.cuda()
        targets = [ann.cuda() for ann in targets]

        # Loss
        loss_l, loss_c = wrapper(images, model, criterion, targets)
        loss = loss_l + loss_c
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
    start = time.perf_counter()
    main()

    wandb.finish()
    print(f"\nExecution time = {time.perf_counter() - start:.2f} second(s)")