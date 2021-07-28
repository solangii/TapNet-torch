import sys
sys.path.append('../')
import argparse

from model.TapNet import TapNet
from utils import str2bool, experiment_name_generator
from data import data_loader

import torch
import torch.nn as nn

def parser_args():
    parser = argparse.ArgumentParser()

    # gpu parameter
    # ------------------------------------------------
    parser.add_argument('--n_gpu', type=int, default=0, help='GPU number to use')
    parser.add_argument('--use_parallel', type=str2bool, default=False, help='Whether to use all GPU')

    # data parameter
    # ------------------------------------------------
    parser.add_argument('--data_root', type=str, default='data/', help='')
    parser.add_argument('--dataset', type=str, default='mini', help='Dataset')

    # Few-shot parameter
    # ------------------------------------------------
    parser.add_argument('--n_shot', type=int, default=1, help='Number of training samples per class')
    parser.add_argument('--n_class_train', type=int, default=20, help='Number of training classes')
    parser.add_argument('--n_class_test', type=int, default=5, help='Number of test classes')
    parser.add_argument('--n_query_train', type=int, default=8, help='Number of queries per class in training')
    parser.add_argument('--n_query_test', type=int, default=15, help='Number of queries per class in test')

    # train parameter
    # ------------------------------------------------
    parser.add_argument('--dim', type=int, default=512, help='Dimension of features')
    parser.add_argument('--n_train_episodes', type=int, default=50000, help = 'Number of train episodes')
    parser.add_argument('--n_val_episodes', type=int, default=600, help='Number of validation episodes')
    parser.add_argument('--n_test_episodes', type=int, default=600, help='Number of test episodes')
    parser.add_argument('--n_test_iter', type=int, default=1, help = 'Iteration number for test')
    parser.add_argument('--meta_batch', type=int, default=1, help='Meta-batch size (number of episodes, but not used in here)')
    parser.add_argument('--wd_rate', type=float, default=5e-4, help='Weight decay rate in Adam optimizer')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--lr_decay', type=int, default=0.1, help='Decaying parameter of learning rate')
    parser.add_argument('--lr_step', type=int, default=40000)

    # save option
    # ------------------------------------------------
    parser.add_argument('--save_root', type=str, default='ckpt/', help='save path for best ckpt')
    parser.add_argument('--info', type=str, default=None, help='Additional notes in the experimental name')

    # test option
    # ------------------------------------------------
    parser.add_argument('--PATH', type=str, default=None, help='Checkpoint path of the model to test')

    # ------------------------------------------------
    config = parser.parse_args()

    config.device = torch.device(f'cuda:{config.n_gpu}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(config.device)  # change allocation of current GPU
    print('Current cuda device ', torch.cuda.current_device())  # check

    return config


def main():
    config = parser_args()
    exp_name = experiment_name_generator(config)
    dataloader = data_loader(config)

    model = TapNet(config, dataloader, exp_name)

    model.train()
    model.test()



if __name__ == '__main__':
    main()