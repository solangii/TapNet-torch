import sys
sys.path.append('../')
import argparse

from model.TapNet import TapNet
from utils import str2bool, experiment_name_generator
from data import data_loader

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

def parser_args():
    parser = argparse.ArgumentParser()

    # data parameter
    parser.add_argument('--data_root', type=str, default='/home/miil/Datasets/TapNet/', help='')
    parser.add_argument('--dataset', type=str, default='mini', help='Dataset')

    # Few-shot parameter
    parser.add_argument('--n_shot', type=int, default=5, help='Number of training samples per class')
    parser.add_argument('--n_class_train', type=int, default=5, help='Number of training classes')
    parser.add_argument('--n_class_test', type=int, default=5, help='Number of test classes')
    parser.add_argument('--n_query_train', type=int, default=8, help='Number of queries per class in training')
    parser.add_argument('--n_query_test', type=int, default=15, help='Number of queries per class in test')

    # train parameter
    parser.add_argument('--dim', type=int, default=512, help='Dimension of features')
    parser.add_argument('--n_episodes', type=int, default=50000, help = 'Number of train episodes')
    parser.add_argument('--wd_rate', type=float, default=5e-4, help='Weight decay rate in Adam optimizer')
    parser.add_argument('--lrdecay', type=str2bool, default=True)
    parser.add_argument('--lrstep', type=int, default=40000)
    parser.add_argument('--batch_size', type=int, default=1)

    # save option
    parser.add_argument('--save_root', type=str, default='save/')

    config = parser.parse_args()
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    return config


def main():
    config = parser_args()
    exp_name = experiment_name_generator(config)
    dataloader = data_loader(config)

    model = TapNet(config, dataloader, exp_name)
    model = model.to(config.device)

    if config.device == 'cuda':
        model = nn.DataParallel(model)
        cudnn.benchmark = True

    model.train()


if __name__ == '__main__':
    main()