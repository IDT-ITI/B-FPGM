# B-FPGM: Lightweight Face Detection via Bayesian-Optimized Soft FPGM Pruning

Bayesian optimised pruning of the lightweight face detector **EResFD**, the currently smallest (in number of parameters) well-performing face detector of the literature; and **EXTD**.

Our methodology divides the network into 6 layer groups optimizes the pruning pruning rate of each group using **Bayesian optimization**. Subsequently, the optimal pruning rate are used in combination with the **Soft Filter Pruning (SFP)** approach and **FPGM pruning**.

<p align="center"><img src="https://github.com/IDT-ITI/B-FPGM/blob/main/Figures/overview.png" alt="drawing" width="1000"/></p>
Overview of our proposed pruning and training pipeline. The diagram on the left illustrates our complete methodology, while the diagram on the right elaborates on the iterative Bayesian optimization process (i.e. the 2nd step of the overall pipeline shown the left). The snowflake symbol indicates that the network's structure remains unchanged during certain steps of the overall pipeline (in contrast to the filter weight values, which are updated throughout all training steps)

## Project Structure

The repository is organized into 4 folders:

- `EResFD-main/`: Contains code and resources for the pre-training and pruning of the EResFD model.
- `EXTD_Pytorch-master`: Contains code and resources for the pre-training and pruning of the EXTD model.
- `Pruned_Models/`: A collection of pruned models (`.pth` files). Pruned models with target pruning rates equal to 10%, 20%, 30%, 40%, 50% and 60% are included, in addition to the original pre-trained and unpruned EResFD and EXTD models.
- `torchscript/`: All the required files for android deployment of the EResFD model (and its pruned versions) using the torchscript framework.

## Installation

Before running the pruning scripts, make sure you have the required dataset and dependencies.

1. **Dataset:** Download and extract the WIDER FACE dataset from [here](https://shuoyang1213.me/WIDERFACE/). Place the `WIDER` folder in the same directory as the project folders.
   
2. **Dependencies:** Python 3.11.9, CUDA 11.4, and Ubuntu 20.04.06 LTS were used in development. Install the required Python dependencies with:

```bash
pip install -r requirements.txt
```


## Training and Pruning
Our methodology consists of 4 sequential steps:

1. Pre-training (`main_pretrain.py`)

2. Bayesian Optimization (`main_bayesian_prune.py`)

3. Soft Filter Pruning (`main_sfp.py`)

4. Fine-tuning (`main_finetune.py`)


The evaluation on the WIDER FACE validation set can be performed by using the script `main_evaluation.py`.

### Important Hyperparameters for Training


| Hyperparameter Name  | Description                                                                              | Default Value |
|----------------------|------------------------------------------------------------------------------------------|---------------|
| `n_epochs_pretraining` | The number of epochs used during the pretraining phase of the model.           | `300`         |
| `n_epochs_sfp`         | The number of epochs used for the Soft Filter Pruning (SFP) process.        | `200`         |
| `n_epochs_finetune`    | The number of epochs used for the fine-tuning stage of the model.          | `10`          |
| `target_pruning_rate`  | The target pruning rate for Bayesian optimization.             | `0.2`         |
| `n_opt_iterations`     | The total number of iterations for Bayesian optimization, after the initial random sampling. | `940`         |
| `n_init_points`        | The number of initial points sampled by the Bayesian optimization. | `60`         |


# Pruned models
In the `Pruned_Models/` folder, we provide pruned models with target pruning rates of 10%, 20%, 30%, 40%, 50% and 60% for EResFD, and 20% for EXTD, along with the original pre-trained and unpruned EResFD and EXTD models. In the following tables, for each model,  we present the pruning rates for each group:


| EResFD Model          | Group 1 Pruning Rate  | Group 2 Pruning Rate  | Group 3 Pruning Rate  | Group 4 Pruning Rate | Group 5 Pruning Rate  | Group 6 Pruning Rate | Sparsity  |
|----------------|--------------------------|--------------------------|--------------------------|--------------------------|--------------------------|--------------------------|--------------|
| Original       | 0%                       | 0%                       | 0%                       | 0%                       | 0%                       | 0%                       | 0            |
| Pruned 10%     | 20%                      | 0%                       | 7.9%                     | 20%                      | 20%                      | 17.2%                    | 10.24%        |
| Pruned 20%     | 0%                       | 0%                       | 0%                       | 40%                      | 40%                      | 39.8%                    | 22.27%        |
| Pruned 30%     | 21.3%                    | 4.5%                     | 30.5%                    | 47.1%                    | 32.1%                    | 44.8%                    | 31.59%        |
| Pruned 40%     | 0.8%                     | 5.2%                     | 47.2%                    | 60%                      | 24.4%                    | 46.4%                    | 40.02%        |
| Pruned 50%     | 0%                       | 0%                       | 38.3%                    | 70%                      | 56.1%                    | 70%                      | 50.37%        |
| Pruned 60%     | 8.2%                     | 0%                       | 80%                      | 80%                      | 80%                      | 16.4%                    | 59.87%        |


| EXTD Model          | Group 1 Pruning Rate  | Group 2 Pruning Rate  | Group 3 Pruning Rate  | Group 4 Pruning Rate | Group 5 Pruning Rate  | Group 6 Pruning Rate | Sparsity  |
|----------------|--------------------------|--------------------------|--------------------------|--------------------------|--------------------------|--------------------------|--------------|
| Original       | 0%                       | 0%                       | 0%                       | 0%                       | 0%                       | 0%                       | 0           |
| Pruned 20%     | 31.1%| 15.7%| 32.4%| 21.6%| 31.1%| 9.8%| 20.3%        |


## Android Deployment

To deploy the pruned EResFD models on Android:
1. Use `convert_to_torchscript.py` to convert the desired model to TorchScript format.
2. Move the generated `lite_scripted_model.ptl` file to the assets folder of your Android application.
3. Load the model in your app using the provided `EResFD.java` class for inference.
A sample `build.gradle.kts` file is included with the necessary dependencies for building the Android app.

## License
This code is provided for academic, non-commercial use only. Please also check for any restrictions applied in the code parts and datasets used here from other sources. For the materials not covered by any such restrictions, redistribution and use in source and binary forms, with or without modification, are permitted for academic non-commercial use provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation provided with the distribution.

This software is provided by the authors "as is" and any express or implied warranties, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed. In no event shall the authors be liable for any direct, indirect, incidental, special, exemplary, or consequential damages (including, but not limited to, procurement of substitute goods or services; loss of use, data, or profits; or business interruption) however caused and on any theory of liability, whether in contract, strict liability, or tort (including negligence or otherwise) arising in any way out of the use of this software, even if advised of the possibility of such damage.

## Citation
If you find our pruning method or pruned models useful in your work, please cite the following publication where this approach was proposed:

N. Kaparinos, V. Mezaris, "B-FPGM: Lightweight Face Detection via Bayesian-Optimized Soft FPGM Pruning", Proc. IEEE/CVF Winter Conference on Applications of Computer Vision Workshops (WACVW 2025), Tucson, AZ, USA, Feb. 2025. 

A pre-print of this paper is available at:  http://arxiv.org/abs/2501.16917

```bibtex
@INPROCEEDINGS{Kaparinos2025,
  author={Kaparinos, Nikolaos and Mezaris, Vasileios},
  booktitle={2025 IEEE/CVF Winter Conference on Applications of Computer Vision Workshops (WACVW)}, 
  title={B-FPGM: Lightweight Face Detection via Bayesian-Optimized Soft FPGM Pruning}, 
  year={2025}
}
```

## Acknowledgement
This work was supported by the EU Horizon Europe and Horizon 2020 programmes under grant agreements 101070093 vera.ai and 951911 AI4Media, respectively.
