from wider_test import eval_wider
from eval_tools.evaluation import *
from utilities import calc_sparsity
import time
import wandb


def main():
    """ Evaluates the model_name and logs the MAPs"""

    # Model inference
    model_name = ''
    net = eval_wider(model_name)

    # Evaluation
    maps = evaluation()
    sparsity = calc_sparsity(net)

    # Log results
    run_name = f'{time.strftime("%b_%d_%Y_%H_%M_%S", time.localtime())}'
    wandb_config = {'Model': model_name}
    run = wandb.init(project='', entity="", name=run_name, config=wandb_config)
    wandb.log({'Easy MAP': maps[0], 'Medium': maps[1], 'Hard': maps[2], 'Sparsity': sparsity})


if __name__ == '__main__':
    main()
