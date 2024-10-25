# B-FPGM: Lightweight Face Detection via Bayesian-Optimized Soft FPGM Pruning

Bayesian optimised pruning of the lightweight face detector **EResFD**, currently the face detector with the lowest number of parameters in the literature.

Our methodology divides the network into 6 layer groups optimizes the pruning pruning rate of each group using **Bayesian optimization**. Subsequently, the optimal pruning rate are used in combination with the **Soft Filter Pruning (SFP)** approach and **FPGM pruning**.

<p align="center"><img src="https://github.com/IDT-ITI/B-FPGM/blob/main/Figures/overview.png" alt="drawing" width="1000"/></p>
Overview of our proposed pruning and training pipeline. The diagram on the left illustrates our complete methodology, while the diagram on the right elaborates on the iterative Bayesian optimization process (i.e. the 2nd step of the overall pipeline shown the left). The snowflake symbol indicates that the network's structure remains unchanged during certain steps of the overall pipeline (in contrast to the filter weight values, which are updated throughout all training steps)

## Project Structure

The repository is organized into 3 folders:

- `EResFD-main/`: Contains code and resources for the pre-training and pruning of the EResFD model.
- `Pruned_Models/`: A collection of pruned models (`.pth` files). The pruned models with target pruning rates equal to 10%, 20%, 30%, 40%, 50% and 60% are included, in addition to the original pre-trained and unpruned EResFD model.
- `torchscript/`: All the required files for android deployment of the EResFD model (and its pruned versions) using the torchscript framework.

## Prerequisites

Before running the pruning scripts, the user needs to prepare the necessary dataset:

### WIDER FACE Dataset

The models are trained and evaluated using the WIDER FACE dataset. To use this dataset:

1. Download the WIDER FACE dataset from [here](https://shuoyang1213.me/WIDERFACE/).
2. Extract and place the `WIDER` folder in the same directory as the `EResFD`, `Pruned models` and `torchscript` folders.

## Dependencies

A requirements.txt file is provided with all the necessary python dependencies. Additionally, the code was developed using Python 3.11.9, CUDA 11.4 and Ubuntu 20.04.06 LTS.

To ensure compatibility and proper functionality of the pruning scripts, please install the specific versions of the python packages listed in the requirements.txt file, using the following command:

```bash
pip install -r requirements.txt
```

## Running the Scripts for Training and Pruning EResFD
Our methodology consists of 4 sequential steps:

1. Pre-training
2. Bayesian Optimization
3. Iterative soft pruning and training
4. Hard pruning and fine-tuning

Each of these steps is implemented in a separate Python script, which can be executed sequentially:
1. `main_pretrain.py`
2. `main_bayesian_prune.py`
3. `main_sfp.py`
4. `main_finetune.py`

, while the evaluation on the WIDER FACE validation set can be performed by using the script `main_evaluation.py`.

Some of the most important hyperparameters and their default values are presented in the table below:

# Hyperparameter Table


| Hyperparameter Name  | Description                                                                              | Default Value |
|----------------------|------------------------------------------------------------------------------------------|---------------|
| `n_epochs_pretraining` | The number of epochs used during the pretraining phase of the model.           | `300`         |
| `n_epochs_sfp`         | The number of epochs used for the Soft Filter Pruning (SFP) process.        | `200`         |
| `n_epochs_finetune`    | The number of epochs used for the fine-tuning stage of the model.          | `10`          |
| `target_pruning_rate`  | The target pruning rate for Bayesian optimization.             | `0.2`         |
| `n_opt_iterations`     | The total number of iterations for Bayesian optimization, after the initial random sampling. | `940`         |
| `n_init_points`        | The number of initial points sampled by the Bayesian optimization. | `60`         |


# Pruned models
In the `Pruned_Models/` folder, we provide pruned models with target pruning rates of 10%, 20%, 30%, 40%, 50%, and 60%, along with the original pre-trained and unpruned EResFD model. In the following table, for each model,  we present the pruning rates for each group:


| Model          | Group 1 Pruning Rate  | Group 2 Pruning Rate  | Group 3 Pruning Rate  | Group 4 Pruning Rate | Group 5 Pruning Rate  | Group 6 Pruning Rate | Sparsity  |
|----------------|--------------------------|--------------------------|--------------------------|--------------------------|--------------------------|--------------------------|--------------|
| Original       | 0%                       | 0%                       | 0%                       | 0%                       | 0%                       | 0%                       | 0            |
| Pruned 10%     | 20%                      | 0%                       | 7.9%                     | 20%                      | 20%                      | 17.2%                    | 11.94%        |
| Pruned 20%     | 0%                       | 0%                       | 0%                       | 40%                      | 40%                      | 39.8%                    | 21.76%        |
| Pruned 30%     | 21.3%                    | 4.5%                     | 30.5%                    | 47.1%                    | 32.1%                    | 44.8%                    | 31.08%        |
| Pruned 40%     | 0.8%                     | 5.2%                     | 47.2%                    | 60%                      | 24.4%                    | 46.4%                    | 39.97%        |
| Pruned 50%     | 0%                       | 0%                       | 38.3%                    | 70%                      | 56.1%                    | 70%                      | 48.88%        |
| Pruned 60%     | 8.2%                     | 0%                       | 80%                      | 80%                      | 80%                      | 16.4%                    | 59.32%        |




## Android Deployment

Simply move the convert_to_torchscript.py script inside the EResFD-main/ directory and run it. Make sure to change the path within the script to convert the desired model to torchscript form.

Subsequently, move the created lite_scripted_model.ptl to your android application`s asset folder. The EResFD.java file contains a java class that can be used in order to load the model from the assets folder and use it for inference.

Additionally, a sample build.gradle.kts file is provided with the necessary dependencies to build the android app.

## License
This code is provided for academic, non-commercial use only. Please also check for any restrictions applied in the code parts and datasets used here from other sources. For the materials not covered by any such restrictions, redistribution and use in source and binary forms, with or without modification, are permitted for academic non-commercial use provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation provided with the distribution. 

This software is provided by the authors "as is" and any express or implied warranties, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed. In no event shall the authors be liable for any direct, indirect, incidental, special, exemplary, or consequential damages (including, but not limited to, procurement of substitute goods or services; loss of use, data, or profits; or business interruption) however caused and on any theory of liability, whether in contract, strict liability, or tort (including negligence or otherwise) arising in any way out of the use of this software, even if advised of the possibility of such damage.

## Citation
[comment]: <If you find our pruning method or pruned models useful in your work, please cite the following publication where this approach was proposed:>
This work was submitted and is under review:

N. Kaparinos, V. Mezaris, "B-FPGM: Lightweight Face Detection via Bayesian-Optimized Soft FPGM
Pruning", under review.
