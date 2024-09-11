"""
B-FPGM
Utilities
CERTH 2024
"""

from typing import Callable, Optional, Tuple, List, Sequence
import numpy as np
import torch
import random
import wandb
import time


def calc_sparsity(model, print_flag:bool =False) -> float:
    """ Calculates the sparsity of the model """
    total_params = 0
    zero_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        zero_params += np.count_nonzero(param.cpu().detach().numpy() == 0)
    sparsity = 100. * zero_params / total_params
    if print_flag:
        print(f"Total params: {total_params}, zero params :{zero_params}, non-zero: {total_params-zero_params}")
        print(f"Sparcity: {sparsity}")
        print("Model sparsity after training: {:.2f}%".format(sparsity))
    return sparsity


def validation(net, criterion, optimizer, epoch: int, config, val_loader, sparsity: float = 0.0,
               min_loss: float = np.inf, save_model: bool = True, save_dir='') -> float:
    """ Validates model by calculating the validation losses"""
    net.eval()
    loc_loss, conf_loss, step = 0, 0, 0
    epoch_running_loss1, epoch_running_loss2 = 0.0, 0.0

    with torch.no_grad():

        for batch_idx, (images, targets) in enumerate(val_loader):
            images = images.cuda()
            targets = [ann.cuda() for ann in targets]

            # Use wrapper() to compute the losses, similar to how you did in the train() function
            t1 = time.time()
            loss_l, loss_c = wrapper(images, net, criterion, targets)
            t2 = time.time()
            print('Timer: %.4f' % (t2 - t1))

            loc_loss += loss_l.item()
            conf_loss += loss_c.item()
            step += 1

            epoch_running_loss1 += loss_l.item()
            epoch_running_loss2 += loss_c.item()

    # Logging
    epoch_running_loss1 /= len(val_loader)
    epoch_running_loss2 /= len(val_loader)
    epoch_val_loss = epoch_running_loss1 + epoch_running_loss2
    wandb.log({'Epoch': epoch, 'Val/Epoch_loss1': epoch_running_loss1, 'Val/Epoch_loss2': epoch_running_loss2,
               'Val/Loss': epoch_val_loss, 'Sparsity': sparsity})

    # Save the model
    if save_model and (epoch_val_loss < min_loss or epoch == config['epochs'] - 1):
        print(f'Saving best state, epoch {epoch}')
        torch.save(net.state_dict(), f'{save_dir}/ERES-{epoch}.pth')
        torch.save(optimizer.state_dict(), f'{save_dir}/optimizerRES.pth')
        min_loss = epoch_val_loss

    return min_loss


def validation_for_optimization(net, criterion, val_loader) -> float:
    """ Calculates the validation losses to be used in the bayesian optimisation """
    net.eval()
    loc_loss, conf_loss, step = 0, 0, 0
    epoch_running_loss1, epoch_running_loss2 = 0.0, 0.0

    with torch.no_grad():
        # t1 = time.time()
        for batch_idx, (images, targets) in enumerate(val_loader):
            images = images.cuda()
            targets = [ann.cuda() for ann in targets]

            # Use wrapper() to compute the losses, similar to how you did in the train() function
            loss_l, loss_c = wrapper(images, net, criterion, targets)

            loc_loss += loss_l.item()
            conf_loss += loss_c.item()
            step += 1

            epoch_running_loss1 += loss_l.item()
            epoch_running_loss2 += loss_c.item()
        # t2 = time.time()
        # print('Timer: %.4f' % (t2 - t1))
    # Logging
    epoch_running_loss1 /= len(val_loader)
    epoch_running_loss2 /= len(val_loader)
    epoch_val_loss = epoch_running_loss1 + epoch_running_loss2
    # wandb.log({'Epoch': epoch, 'Val/Epoch_loss1': epoch_running_loss1, 'Val/Epoch_loss2': epoch_running_loss2,
    #            'Val/Loss': epoch_val_loss, 'Sparsity': sparsity})
    return epoch_val_loss

def wrapper(images, net, criterion, targets):
    outputs = net(images)
    total_loss_l = 0
    total_loss_c = 0
    for out in outputs:
        loss_l, loss_c = criterion(out, targets)
        total_loss_l += loss_l
        total_loss_c += loss_c
    return total_loss_l, total_loss_c


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def count(model):
    total_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()

    print(f'Num params = {total_params}')


def compute_flops(model, image_size):
    import torch.nn as nn
    flops = 0.
    input_size = image_size
    for m in model.modules():
        if isinstance(m, nn.AvgPool2d) or isinstance(m, nn.MaxPool2d):
            input_size = input_size / 2.
        if isinstance(m, nn.Conv2d):
            if m.groups == 1:
                flop = (input_size[0] / m.stride[0] * input_size[1] / m.stride[1]) * m.kernel_size[
                    0] ** 2 * m.in_channels * m.out_channels
            else:
                flop = (input_size[0] / m.stride[0] * input_size[1] / m.stride[1]) * m.kernel_size[0] ** 2 * (
                        (m.in_channels / m.groups) * (m.out_channels / m.groups) * m.groups)
            flops += flop
            if m.stride[0] == 2: input_size = input_size / 2.

    return flops / 1000000000., flops / 1000000


def set_all_seeds(seed: Optional[int] = 0) -> None:
    """ Set all seeds """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True  # noqa
