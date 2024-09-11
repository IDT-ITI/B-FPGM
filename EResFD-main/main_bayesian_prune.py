"""
B-FPGM
Run the Bayesian optimization for the B-FPGM pruner
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
import torch.optim as optim
import torch.utils.data as data
from nni.algorithms.compression.v2.pytorch.pruning import FPGMPruner, L1NormPruner
from prepare_wider_data import wider_data_file
from data.config import cfg
from models.eresfd import build_model
from layers.modules.multibox_loss import MultiBoxLoss
from data.factory import dataset_factory, detection_collate
from bayesian_pruner import *
from utilities import *
from copy import deepcopy
from wider_test import eval_wider
from eval_tools.evaluation import *
import yaml
import pickle


# compile
os.system("python3 bbox_setup.py build_ext --inplace")
print('compile completed')

def main():
    """ Trains and prunes the model iteratively """
    pruning_hparams = {'target_pruning_rate': 0.2, 'n_opt_iterations': 940, 'base_pruner': 'FPGM',
                       'probing_offset': 0.04, 'regularization_coefficient': 5, 'n_init_points': 60, 'seed': 0}
    config = {**pruning_hparams,
              'pre_trained_model': ''}
    set_all_seeds(config['seed'])

    # Init dataset
    print('prepare wider')
    wider_data_file()

    # Datasets and dataloaders
    train_dataset, test_dataset = dataset_factory('face')
    n_train = int(0.8 * len(train_dataset))
    n_test = len(train_dataset) - n_train
    train_dataset2, val_dataset = torch.utils.data.random_split(train_dataset, [n_train, n_test],
                                                                generator=torch.Generator(device='cuda'))
    val_dataset_small = data.Subset(val_dataset, [i for i in range(100)])
    train_loader = data.DataLoader(train_dataset2, 16, num_workers=0, shuffle=True,
                                   collate_fn=detection_collate, pin_memory=False,
                                   generator=torch.Generator(device='cuda'))
    val_loader = data.DataLoader(val_dataset, 16, num_workers=0, shuffle=False,
                                 collate_fn=detection_collate, pin_memory=False)

    # Init model, criterion, and optimizer
    net = build_model('test', cfg.NUM_CLASSES, width_mult=0.0625)
    print('Load network....')
    net.load_state_dict(torch.load(config['pre_trained_model']))
    print('Network loaded successfully')

    # define loss function (criterion) and optimizer
    criterion = MultiBoxLoss(cfg, 'face', True)
    net.cuda()


    # 4 Pruning Groups
    # pruning_groups = [["base.conv1", "base.conv2", "base.conv3", "base.conv4"],
    #                   ["base.m0.layer1", "base.m0.layer2", "base.m0.layer3", "base.m0.layer4", "base.m0.layer5"],
    #                   ["base.m0.fpn"],
    #                   ["base.m0.layer7", "base.m0.layer8", "base.m0.layer9", "base.m0.layer10", "base.m0.layer11",
    #                    "base.m0.layer12"]
    #                   ]

    # 6 Pruning Groups
    pruning_groups = [["base.conv1", "base.conv2", "base.conv3"],
                      ["base.conv4", "base.m0.layer1"],
                      ["base.m0.layer2", "base.m0.layer3"],
                      ["base.m0.layer4", "base.m0.layer5"],
                      ["base.m0.fpn"],
                      ["base.m0.layer7", "base.m0.layer8", "base.m0.layer9", "base.m0.layer10", "base.m0.layer11",
                       "base.m0.layer12"]
                      ]

    # 9 Pruning Groups
    # pruning_groups = [["base.conv1", "base.conv2", "base.conv3"],
    #                   ["base.conv4"],
    #                   ["base.m0.layer1"],
    #                   ["base.m0.layer2"],
    #                   ["base.m0.layer3"],
    #                   ["base.m0.layer4"],
    #                   ["base.m0.layer5"],
    #                   ["base.m0.fpn"],
    #                   ["base.m0.layer7", "base.m0.layer8", "base.m0.layer9", "base.m0.layer10", "base.m0.layer11",
    #                    "base.m0.layer12"]
    # ]
    net.phase = "train"
    bayesian_pruner = BayesianPruner(pruning_groups=pruning_groups, val_loader=val_loader, train_loader=train_loader,
                                     criterion=criterion, **pruning_hparams) # noqa
    optimal_pruning_dict = bayesian_pruner.prune(net)


    # Save results
    log_dir = bayesian_pruner.log_dir

    with open(f'{log_dir}/pruning_config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    with open(f'{log_dir}/pruning_groups.yaml', 'w') as f:
        yaml.dump(pruning_groups, f, default_flow_style=False)
    optimal_pruning_rates = list(optimal_pruning_dict.values())
    print(f'{optimal_pruning_rates = }')
    with open(f'{log_dir}/optimal_pruning_rates.pkl', 'wb') as f:
        pickle.dump(optimal_pruning_rates, f)



if __name__ == '__main__':
    start = time.perf_counter()
    main()

    wandb.finish()
    print(f"\nExecution time = {time.perf_counter() - start:.2f} second(s)")