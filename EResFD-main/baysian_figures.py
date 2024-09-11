"""
B-FPGM
Creates and saves figures from the Bayesian optimization results
CERTH 2024
"""

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def calc_cumulative_min(validation_loss: pd.Series, objective_fn_value: pd.Series):
    """ Calculates the cumulative min for the validation loss
    but does not take account samples with objective_fn_value > 100
    """
    data = []
    min_loss = np.inf
    for i in range(len(validation_loss)):
        if (objective_fn_value[i] > -100) and (validation_loss[i] < min_loss):
            min_loss = validation_loss[i]
        data.append(min_loss)
    validation_cummin = pd.Series(data=data)
    return validation_cummin


def main():
    """ Creates and saves figures from the bayesian optimization results"""
    opt_log_dir = ('/LOG_DIR'
                   '/RUN')
    results_file = f'{opt_log_dir}/optimization_results.csv'

    # Read csv file
    results = pd.read_csv(results_file)
    validation_loss = results['val_loss']
    objective_fn_value = results['objective_fn']
    validation_cummin = calc_cumulative_min(validation_loss, objective_fn_value)
    objective_fn_cummax = objective_fn_value.cummax()

    # Figures
    # Validation loss
    sns.set()
    fig = plt.figure(0, figsize=(12, 8))
    sns.scatterplot(data=validation_loss)
    plt.ylabel('Validation loss')
    plt.xlabel('Optimization iteration')
    plt.title('Validation loss throughout the optimization', size=16)
    fig.savefig(f'{opt_log_dir}/validation_loss.png', dpi=150)

    fig = plt.figure(1, figsize=(12, 8))
    sns.lineplot(data=validation_cummin)
    plt.ylabel('Validation loss')
    plt.xlabel('Optimization iteration')
    plt.title('Min validation loss throughout the optimization', size=16)
    fig.savefig(f'{opt_log_dir}/validation_loss_min.png', dpi=150)

    # Objective function
    fig = plt.figure(2, figsize=(12, 8))
    sns.scatterplot(data=objective_fn_value)
    plt.ylabel('Objective function value')
    plt.xlabel('Optimization iteration')
    plt.title('Objective function value throughout the optimization', size=16)
    fig.savefig(f'{opt_log_dir}/objective_fn.png', dpi=150)

    fig = plt.figure(3, figsize=(12, 8))
    sns.lineplot(data=objective_fn_cummax)
    plt.ylabel('Objective function value')
    plt.xlabel('Optimization iteration')
    plt.title('Max objective function value throughout the optimization', size=16)
    fig.savefig(f'{opt_log_dir}/objective_fn_value_max.png', dpi=150)


if __name__ == '__main__':
    main()
