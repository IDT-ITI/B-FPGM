# B-FPGM: Lightweight Face Detection via Bayesian-Optimized Soft FPGM Pruning

Bayesian optimised pruning of the lightweight face detector **EResFD**, currently the face detector with the lowest number of parameters in the literature.

Our methodology divides the network into 6 layer groups optimizes the pruning pruning rate of each group using **Bayesian optimization**. Subsequently, the optimal pruning rate are using in combination with the **Soft Filter Pruning (SFP)** approach and **FPGM pruning**.

## Project Structure

The repository is organized into 4 folders:

- `EResFD-main/`: Contains code and resources for the pre-training and pruniong of the EResFD model.
- `Pruned_Models/`: A collection of pre-pruned model weights (`.pth` files). The pruned models with target pruning rates equal to 10%, 20%, 30%, 40%, 50% and 60% are included, in addition to the original pre-trained and unpruned EResFD model.
- `torchscript/`: All the required files for android deployment of the EResFD model (and its pruned versions) using the torchscript framework.

## Prerequisites

Before running the pruning scripts, the user needs to prepare the necessary dataset:

### WIDER FACE Dataset

The models are trained and evaluated using the WIDER FACE dataset. To use this dataset:

1. Download the WIDER FACE dataset from [here](https://shuoyang1213.me/WIDERFACE/).
2. Extract and place the `WIDER` folder in the same directory as the `EXTD` and `EResFD` folders.

## Dependencies

A requirements.txt file is provided with all the necessary python dependencies. Additionally, the code was developed using Python 3.11.7, CUDA 11.4 and Ubuntu 20.04.06 LTS.

To ensure compatibility and proper functionality of the pruning scripts, please install the specific versions of the python packages listed in the requirements.txt file, using the following command:

```bash
pip install -r requirements.txt
```
