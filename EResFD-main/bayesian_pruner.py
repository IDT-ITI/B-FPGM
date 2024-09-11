"""
B-FPGM
B-FGPM pruner class
CERTH 2024
"""

import pickle

import numpy as np
import pandas as pd
import yaml
from bayes_opt import BayesianOptimization

from nni.algorithms.compression.v2.pytorch.pruning import FPGMPruner, L1NormPruner
from torch import optim
from typing import Callable, Optional, Tuple, List, Sequence, Dict
from utilities import validation_for_optimization, calc_sparsity
from copy import deepcopy
from os import makedirs
from utilities import wrapper
import time

supported_base_pruners = ("FPGM", "l1")


def calc_pruning_penalty(target_pruning: float, actual_pruning: float):
    """ Penalty term for the pruning rate """
    difference = np.abs(actual_pruning - target_pruning)
    threshold = 0.04
    if difference >= threshold:
        penalty_value = 100
    elif target_pruning <= actual_pruning <= target_pruning + threshold:
        penalty_value = 0
    elif actual_pruning >= target_pruning - threshold:
        penalty_value = difference
    else:
        penalty_value = 0
    return penalty_value


def calc_objective_fn_value(val_loss, target_pruning, actual_pruning, regularization_coefficient):
    """ Calculates of the optimization objective fn value"""
    return -val_loss - regularization_coefficient * calc_pruning_penalty(target_pruning, actual_pruning)


def get_nested_attribute(obj, attribute_string):
    attributes = attribute_string.split('.')
    for attr in attributes:
        obj = getattr(obj, attr)
    return obj


class MultiLayerPruner:
    """ Prunes multiple layers using a different pruning rate per layer """

    def __init__(self, pruning_dict: Dict[Tuple[str], float], pruner: str = "FPGM"):
        if pruner == 'l1':
            self.pruner = L1NormPruner
        elif pruner == 'FPGM':
            self.pruner = FPGMPruner
        else:
            raise ValueError('Pruner not supported')

        self.pruning_dict = pruning_dict
        # example: pruning_dict = {tuple(["base.conv1", "base.conv2"]): 0.3}

    def prune(self, model):
        for group, pruning_rate in self.pruning_dict.items():
            for layer in group:
                config_list = [{
                    'sparsity_per_layer': pruning_rate,
                    'op_types': ['Conv2d'],
                }]
                pruner = self.pruner(get_nested_attribute(model, layer), config_list)
                pruner.compress()
                pruner._unwrap_model()


class BayesianPruner:
    """ Pruner that utilizes bayesian optimization to optimize the pruning rate of each layer group """

    def __init__(self, target_pruning_rate: float, pruning_groups: Tuple[Tuple[str]], val_loader, train_loader,
                 criterion, n_opt_iterations: int = 5, probing_offset: float = 0.02,
                 regularization_coefficient: float = 5.0, n_init_points: int = 4, base_pruner: str = "FPGM",
                 seed: int = 0):
        assert base_pruner in supported_base_pruners, f"Base pruner {base_pruner} not supported!"
        self.target_pruning_rate = target_pruning_rate
        self.pruning_groups = pruning_groups
        self.n_variables = len(self.pruning_groups)
        self.probing_offset = probing_offset
        self.regularization_coefficient = regularization_coefficient
        self.val_loader = val_loader
        self.train_loader = train_loader
        self.criterion = criterion
        self.n_opt_iterations = n_opt_iterations
        self.n_init_points = n_init_points
        self.base_pruner = base_pruner
        self.seed = seed
        self.log_dir = None

    def prune(self, model):
        objective_fn = self.create_objective_fn(model=model)
        pbounds = {f"x{i}": (0, min(0.98, self.target_pruning_rate + 0.2)) for i in range(self.n_variables)}

        optimizer = BayesianOptimization(f=objective_fn, pbounds=pbounds, verbose=2, random_state=self.seed,
                                         allow_duplicate_points=True)
        optimizer.probe(params=[self.target_pruning_rate + self.probing_offset] * self.n_variables, lazy=True)

        # Optimize
        start = time.perf_counter()
        optimizer.maximize(init_points=self.n_init_points, n_iter=self.n_opt_iterations)

        optimal_pruning_rates = list(optimizer.max['params'].values())
        pruning_dict = {tuple(group): group_pruning_rate for group, group_pruning_rate in
                        zip(self.pruning_groups, optimal_pruning_rates)}
        layer_pruner = MultiLayerPruner(pruning_dict, pruner=self.base_pruner)
        layer_pruner.prune(model)

        print(f"\nExecution time = {time.perf_counter() - start:.2f} second(s)")
        print(optimizer.max)

        return pruning_dict

    def create_objective_fn(self, model) -> callable:
        """ Creates and returns the optimisation objective fn"""
        optimization_run_name = f'{time.strftime("%b_%d_%Y_%H_%M_%S", time.localtime())}'
        self.log_dir = f'logs/bayesian_opt/{optimization_run_name}'
        makedirs(self.log_dir)
        global opt_results
        columns = ["iter"] + [f"x{i}" for i in range(self.n_variables)] + ["objective_fn", "val_loss",
                                                                           "pruning_penalty", "sparsity"]
        opt_results = pd.DataFrame(columns=columns)

        def obj_fn(**kwargs):
            # Multilayer pruner
            pruning_dict = {tuple(group): value for group, (variable, value) in
                            zip(self.pruning_groups, kwargs.items())}
            model_cpy = deepcopy(model)
            layer_pruner = MultiLayerPruner(pruning_dict)
            layer_pruner.prune(model_cpy)
            actual_sparsity = calc_sparsity(model_cpy) / 100

            pruning_penalty = calc_pruning_penalty(self.target_pruning_rate, actual_sparsity)

            # Train
            if pruning_penalty < 10:
                optimizer = optim.Adam(model_cpy.parameters(), lr=1e-3, weight_decay=1e-5)
                model_cpy.train()
                train(self.train_loader, model_cpy, self.criterion, optimizer)

                # Validation
                val_loss = validation_for_optimization(model_cpy, self.criterion, self.val_loader)
            else:
                val_loss = 0

            # Calculate objective fn value
            objective_fn_value = calc_objective_fn_value(val_loss, self.target_pruning_rate, actual_sparsity,
                                                         self.regularization_coefficient)
            i = len(opt_results)
            new_row = [i, *kwargs.values(), objective_fn_value, val_loss, pruning_penalty, actual_sparsity]
            opt_results.loc[len(opt_results)] = new_row
            opt_results.to_csv(f"{self.log_dir}/optimization_results.csv", index=False)

            return objective_fn_value

        return obj_fn


def train(train_loader, model, criterion, optimizer):
    model.train()
    for batch_idx, (images, targets) in enumerate(train_loader):
        images = images.cuda()
        targets = [ann.cuda() for ann in targets]

        # Loss
        optimizer.zero_grad()
        loss_l, loss_c = wrapper(images, model, criterion, targets)
        loss = loss_l + loss_c
        loss.backward()
        optimizer.step()


def load_pruning_dict(log_dir: str) -> Dict:
    with open(f'{log_dir}/pruning_groups.yaml', 'r') as f:
        yaml.safe_load(f)
    with open(f'{log_dir}/optimal_pruning_rates.pkl', 'rb') as f:
        optimal_pruning_rates = pickle.load(f)
