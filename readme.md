# TapNet: Neural Network Augmented with Task-Adaptive Projection for Few-Shot Learning

An unofficial Code for ICML 2019 paper TapNet: Neural Network Augmented with Task-Adaptive Projection for Few-Shot Learning in PyTorch. [![arXiv](https://img.shields.io/badge/arXiv-1905.06549-f9f107.svg)](https://arxiv.org/abs/1905.06549)

Original Code (Chainer) : https://github.com/istarjun/TapNet

## Dependencies

```
python == 3.6.13
torch == 1.8.1
torchmeta == 1.7.0
cupy == 9.2.0
```

### Conda enviornment

```
conda env create -f TapNet-torch.yaml
source activate TapNet
```

### pip dependency

```
pip install -r requirements.txt
```

## How to run

```
python main.py 
```

### Usage

```
usage: main.py [-h] [--n_gpu N_GPU] [--use_parallel USE_PARALLEL]
               [--data_root DATA_ROOT] [--dataset DATASET] [--n_shot N_SHOT]
               [--n_class_train N_CLASS_TRAIN] [--n_class_test N_CLASS_TEST]
               [--n_query_train N_QUERY_TRAIN] [--n_query_test N_QUERY_TEST]
               [--dim DIM] [--n_train_episodes N_TRAIN_EPISODES]
               [--n_val_episodes N_VAL_EPISODES] [--meta_batch META_BATCH]
               [--wd_rate WD_RATE] [--lr LR] [--lr_decay LR_DECAY]
               [--lr_step LR_STEP] [--save_root SAVE_ROOT] [--info INFO]

optional arguments:
  -h, --help            show this help message and exit
  --n_gpu N_GPU         GPU number to use
  --use_parallel USE_PARALLEL
                        Whether to use all GPU
  --data_root DATA_ROOT
  --dataset DATASET     Dataset
  --n_shot N_SHOT       Number of training samples per class
  --n_class_train N_CLASS_TRAIN
                        Number of training classes
  --n_class_test N_CLASS_TEST
                        Number of test classes
  --n_query_train N_QUERY_TRAIN
                        Number of queries per class in training
  --n_query_test N_QUERY_TEST
                        Number of queries per class in test
  --dim DIM             Dimension of features
  --n_train_episodes N_TRAIN_EPISODES
                        Number of train episodes
  --n_val_episodes N_VAL_EPISODES
                        Number of validation episodes
  --meta_batch META_BATCH
                        Meta-batch size (number of episodes, but not used in
                        here)
  --wd_rate WD_RATE     Weight decay rate in Adam optimizer
  --lr LR               Learning rate
  --lr_decay LR_DECAY   Use weight decay
  --lr_step LR_STEP
  --save_root SAVE_ROOT
                        save path for best ckpt
  --info INFO           Additional notes in the experimental name
```

See function `parser_args()` in `main.py` file for **detailed optional arguments**.

## Result



