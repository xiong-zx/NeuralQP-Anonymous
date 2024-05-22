# NeuralQP-Anonymous

This repository contains the code for `NeuralQP`, a general hypergraph-based framework for large-scale Quadratically Constrained Quadratic Programs (QCQPs). This document provides detailed instructions for setting up the environment, installing dependencies, and downloading the dataset.

## Installation

### 1. Create a New Conda Environment

We recommend using Conda to manage your environment. You can create a new environment. We recommend using Python 3.10.

```bash
conda create -n [env] python=3.10
conda activate [env]
```

### 2. Install Gurobi and SCIP

At least one of the solvers (Gurobi or SCIP) is required for the experiments. Install both to run all experiments.

#### Gurobi

Gurobi is a commercial optimization solver that can be used for solving optimization problems. It is free for academic use. To use Gurobi in this project, follow the steps below:

1. Download and install Gurobi software from [gurobi download page](https://www.gurobi.com/downloads/gurobi-software/) according to your machine.
2. Obtain a license and follow the installation instructions.
3. Install the python interface for Gurobi using `pip install gurobipy`.

#### SCIP

We recommend installing SCIP on Linux. Follow the instructions [here](https://scipopt.org/doc/html/INSTALL.php) to install SCIP.

### 3. Install Dependencies

Since different systems may have different CUDA versions, we recommend using `pip` to install the following dependencies manually.

```bash
dhg
torch
torch_geometric
numpy
pandas
matplotlib
tqdm
pyscipopt
gurobipy
scikit-learn
hydra-core
ipykernel # for jupyter notebook
```

Due to compatibility issues, specific versions of `torch` and `torch_geometric` are required. We recommend using `torch` version 1.13 for compatibility with `dhg`. Additionally, you need to install these libraries based on your CUDA version.

1. Install [`dhg`](https://deephypergraph.readthedocs.io/en/latest/start/install.html) using `pip install dhg`. This might help you install the required `torch` dependencies.
2. Install [`torch_geometric`](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) by `pip install torch_geometric`
3. Install dependencies of `torch_geometric` based on your CUDA version.

```bash
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
```

If other dependencies are missing, please install them using `pip`.

### 4. Download the Dataset

Download the [dataset](https://drive.google.com/file/d/1rdhoOrzsEp2XQULx9bR_Txo3zFzLWPkn/view?usp=drive_link) and place it in the root directory of the project.

```bash
tar -xzvf data.tar.gz
mv _data/ data/
rm data.tar.gz
```

## Running the Code

```bash
cd core
```

### 1. Encode the Problems into Hypergraphs

```bash
python Encoding.py -p [problem_name] -d [difficulty]
```

- `problem_name`: the name of the problem to be encoded. `qis` for RandQCP, `qkp` for QMKP.
- `difficulty`: the difficulty level of the problem. Must be `tiny`, `easy` or `medium`.

> **Attention**
>
> In this repository, we use different aliases for problem scales.
>
> - `tiny`: Tiny in our paper
> - `easy`: 1000 in our paper
> - `medium`: 2000 in our paper
> - `5000`: 50000 in our paper
> - `10000`: 10000 in our paper

### 2. Train a UniEGNN Model

```bash
python NeuralPrediction.py -p [problem_name] -d [difficulty]
```

- `problem_name`: the name of the problem to be trained.
- `difficulty`: the difficulty level of the problem.

Additional command line arguments:

- `--nlayer`: Number of layers in the model. Default is 6.
- `--nout`: Number of output features. Default is 1. Choices: [1].
- `--nhid`: Number of hidden units. Default is 64.
- `--drop_rate`: Dropout rate. Default is 0.1.
- `--input_drop`: Input dropout rate. Default is 0.0.
- `--first_aggregate`: The first aggregation method. Default is 'sum'. Choices: ['sum', 'mean', 'softmax_then_sum'].
- `--second_aggregate`: The second aggregation method. Default is 'mean'. Choices: ['sum', 'mean', 'softmax_then_sum'].
- `--nobias`: If set, no bias is used. Default is False.
- `--fix_edge`, `-f`: If set, the edge features are fixed. Default is False.
- `--activation`: Activation function. Default is 'leakyrelu'.
- `--lr`: Learning rate. Default is 1e-4.
- `--wd`: Weight decay. Default is 1e-4.
- `--nepoch`: Number of epochs for training. Default is 100.
- `--early_stop`: Number of epochs for early stopping. Default is 100.
- `--batch_size`: Batch size for training. Default is 16.
- `--seed`: Random seed. Default is 42.
- `--device`: Device to be used for training. Default is 'cuda:0'.
- `--loss`: Loss function. Default is 'bce'. Choices: ['mse', 'bce', 'focal'].
- `--visualize`, `-v`: Visualize the neural outputs of the given problem. Specify the **problem name with no suffix** to visualize (e.g., `qis_1000_800_5_0` ).
- `--vepoch`: Epoch for visualization. Default is 0. Use -1 for default epochs in separate figures, 0 for default epochs in one figure, or an integer for a specific epoch.
- `--no_train`, `-n`: If set, the model is not trained. Default is False.
- `--output`, `-o`: If set, outputs the neural outputs. Default is False.

> **Attention**
>
> 1. We do not provide pretrained models since they might not work on different systems. You need to put the model in `models/` and rename it as `[problem]_[difficulty].pkl`, e.g., `qis_medium.pkl`.
> 2. You need to change the `model` arguments in `config/*.yaml` to match the model you have trained.

### 3. Main Experiments

#### 3.1 Run the Baseline

```bash
python Baseline.py
```

You need to modify the arguments in `config/baseline.yaml`. Here's a brief explanation for necessary arguments in the `baseline.yaml` file:

- `run`
  - `solver`: Specifies the solver to be used. Options are `gurobi` or `scip`.
  - `problem`: Specifies the problem type. Options are `qis` or `qkp`.
  - `difficulty`: Sets the scale of the problem. Options include `tiny`, `easy`, `medium`, `5000`, `10000`.
  - `time_limit`: Sets the time limit for the solver in seconds.
  - `gap_limit`: No specific gap limit is set.
  - `n_instances`: Specifies the number of problem instances to run.

#### 3.2 Run NeuralQP

```bash
python Main.py
```

You need to modify the arguments in `config/main.yaml`. Here's a brief explanation for necessary arguments in the `main.yaml` file:

- `run`
  - `seed`: Sets the random seed for reproducibility.
  - `n_instances`: Specifies the number of problem instances to run.
  - `train`
    - `problem`: Specifies the problem type for training. Options include `qis` or `qkp`.
    - `difficulty`: Sets the scale of the problem for training. Options include `tiny`, `easy`, `medium`, `5000`, `10000`.
  - `test`
    - `problem`: Specifies the problem type for testing. Options include `qis` or `qkp`.
    - `difficulty`: Sets the scale of the problem for testing. Options include `tiny`, `easy`, `medium`, `5000`, `10000`.
  - `device`: Specifies the device to be used for training and testing.

- `lns`
  - `solver`: Specifies the solver to be used for LNS. Options include `gurobi` or `scip`.
  - `n_times`: Number of times to run the LNS.
  - `n_processes`: Number of parallel processes to use.
  - `block_size`: Specifies the block size for the LNS. Must be a float between 0 and 1.
  - `obj_limit`: Sets the objective limit for the LNS.
  - `time_limit`: Sets the time limit for the LNS in seconds.
  - `policies`
    - `crossover`: Whether to use crossover in LNS. Options are `True` or `False`.
    - `neighborhood_policy`: Policy for selecting neighborhoods. Options are `random` or `constr_random`.
    - `repair_policy`: Policy for repairing solutions. Options are `quick` or `cautious`.

To analyze the results, you can run

```bash
python analyzeMain.py
```

Make sure that there exists a `results` directory in the root directory.

### 4. Extra Experiments

#### 4.1 Test Neural Prediction

```bash
python testNP.py
```

You need to modify the arguments in `config/NP.yaml`. Here's a brief explanation for necessary arguments in the `NP.yaml` file:

- `run`
  - `seed`: Sets the random seed for reproducibility. Default is 42.
  - `train`
    - `problem`: Specifies the problem type for training. Options include `qis` or `qkp`.
    - `difficulty`: Sets the scale of the problem for training. Options include `tiny`, `easy`, `medium`, `5000`, `10000`.
  - `test`
    - `problem`: Specifies the problem type for testing. Options include `qis` or `qkp`.
    - `difficulty`: Sets the scale of the problem for testing. Options include `tiny`, `easy`, `medium`, `5000`, `10000`.
  - `device`: Specifies the device to be used for training and testing.

#### 4.2 Test Parallel Neighborhood Optimization

```bash
python testLNS.py
```

You need to modify the arguments in `config/LNS.yaml`. Here's a brief explanation for necessary arguments in the `LNS.yaml` file:

- `run`
  - `seed`: Sets the random seed for reproducibility.
  - `problem`: Specifies the problem type. Options are `qis` or `qkp`.
  - `difficulty`: Sets the scale of the problem. Options include `tiny`, `easy`, `medium`, `5000`, `10000`.
  - `method`: Specifies the method to be used. Options are `lns`, `gurobi`, or `scip`.
  - `n_solutions`: Specifies the number of initial solutions to.
- `lns`
  - `solver`: Specifies the solver to be used for LNS. Options include `scip` or `gurobi`.
  - `ntimes`: Number of times to run the LNS.
  - `nprocesses`: Number of parallel processes to use.
  - `block_size`: Specifies the block size for the LNS. Must be a float between 0 and 1.
  - `time_limit`
    - `cross_time_limit`: Time limit for the crossover phase in seconds.
    - `search_time_limit`: Time limit for the search phase in seconds.
  - `policies`
    - `crossover`: Whether to use crossover in LNS. Options are `True` or `False`.
    - `neighborhood_policy`: Policy for selecting neighborhoods. Options are `random` or `constr_random`.
    - `repair_policy`: Policy for repairing solutions. Options are `quick` or `cautious`.