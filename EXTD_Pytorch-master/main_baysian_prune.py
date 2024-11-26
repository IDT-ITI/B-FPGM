'''
EXTD Copyright (c) 2019-present NAVER Corp. MIT License
'''

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
from EXTD_64 import build_extd
from layers.modules.multibox_loss import MultiBoxLoss
from data.factory import dataset_factory, detection_collate
from os import makedirs
from tqdm import tqdm
from bayesian_pruner import *
from utilities import *
from copy import deepcopy
from wider_test import eval_wider
from eval_tools.evaluation import *
from utilities import calc_sparsity
import yaml
import pickle
import json
import ujson

# print("sleeping...")
# time.sleep(1.6 * 60 * 60)
# print("woke up!")

# compile
os.system("python3 bbox_setup.py build_ext --inplace")


# print('compile completed')

def main():
    """ Trains and prunes the model iteratively """
    pruning_hparams = {'target_pruning_rate': 0.5, 'n_opt_iterations': 940, 'base_pruner': 'FPGM',
                       'probing_offset': 0.04, 'pruning_penalty_lagrange_mult': 1.0, 'difference_multiplier': 5,
                       'n_init_points': 60, 'seed': 0}
    config = {**pruning_hparams,
              'pre_trained_model': '/home/kaparinos/Projects/Lightweight-Face-Detector-Internal/EXTD_Pytorch-master/weights/default_train_script_2/sfd_default_script_300_checkpoint.pth'}
    set_all_seeds(config['seed'])

    # Init dataset
    print('prepare wider')
    wider_data_file()

    # Datasets and dataloaders
    # train_dataset, val_dataset = dataset_factory('face')
    train_dataset, test_dataset = dataset_factory('face')
    n_train = int(0.8 * len(train_dataset))
    n_test = len(train_dataset) - n_train
    train_dataset2, val_dataset = torch.utils.data.random_split(train_dataset, [n_train, n_test],
                                                                generator=torch.Generator(device='cuda'))
    train_loader = data.DataLoader(train_dataset2, 4, num_workers=0, shuffle=True,
                                   collate_fn=detection_collate, pin_memory=False,
                                   generator=torch.Generator(device='cuda'))
    val_loader = data.DataLoader(val_dataset, 4, num_workers=0, shuffle=False,
                                 collate_fn=detection_collate, pin_memory=False)

    # Init model, criterion, and optimizer
    s3fd_net = build_extd('train', cfg.NUM_CLASSES)
    net = s3fd_net
    print('Load network....')
    net.base.load_state_dict(torch.load(config['pre_trained_model']), strict=False)
    print('Network loaded successfully')

    # define loss function (criterion) and optimizer
    criterion = MultiBoxLoss(cfg, 'face', True)
    net.cuda()

    pruning_groups = [["base.0", "base.1"],
                      ["base.2", "base.3"],
                      ["base.4"],
                      ["base.5"],
                      ["upfeat"],
                      ["loc", "conf"]
                      ]
    net.phase = "train"
    bayesian_pruner = BayesianPruner(pruning_groups=pruning_groups, val_loader=val_loader, train_loader=train_loader,
                                     criterion=criterion, **pruning_hparams)  # noqa
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

    with open(f'{log_dir}/optimal_pruning_rates.pkl', 'rb') as f:
        optimal_pruning_rates2 = pickle.load(f)
    print(f'{optimal_pruning_rates2 = }')


if __name__ == '__main__':
    start = time.perf_counter()
    main()

    print(f"\nExecution time = {time.perf_counter() - start:.2f} second(s)")
