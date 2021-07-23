import os
import sys
sys.path.append('../')
import argparse

import numpy as np

from model import TapNet
from utils import str2bool, experiment_name_generator
from data import data_loader

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

def parser_args():
    parser = argparse.ArgumentParser()

    # data parameter
    parser.add_argument('--data_root', type=str, default='data/', help='')
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

def train(model, config, dataloader, exp_name):
    loss_h = []
    accuracy_h_val = []
    accuracy_h_test = []

    acc_best = 0
    epoch_best = 0

    for idx, episode in enumerate(dataloader['meta_train']):
        support_data, support_label = episode['train']
        query_data, query_label = episode['test']
        support_data, support_label = support_data.to(config.device), support_label.to(config.device)
        query_data, query_label = query_data.to(config.device), query_label.to(config.device)

        loss = model.train(support_data, support_label) # 맞나

        # logging
        # --------------------------------
        loss_h.extend([loss.tolist()])
        if idx % 50 ==0:
            print("Episode: %d, Train Loss: %f "%(idx, loss))

        if idx!=0 and idx%500 ==0:
            print("Evaluation in Validation data")
            scores = []

            for idx, episode in enumerate(dataloader['meta_val']): #몇개 돌지 정해두기
                accs = model.evaludate(support_data, support_label)
                accs_ = [cuda.to_cpu(acc) for acc in accs] # 이거머지
                score = np.asarray(accs_, dtype=int) # 이거머지
                scores.append(score)

            print(('Accuracy 5 shot ={:.2f}%').format(100*np.mean(np.array(scores))))
            accuracy_t = 100*np.mean(np.array(scores))

            if acc_best < accuracy_t:
                acc_best = accuracy_t
                epoch_best = idx
                # save model Todo
                # 뭐시l save npz

            accuracy_h_val.extend([accuracy_t.tolist()])
            del(accs) # 이거머지
            del(accs_)
            del(accuracy_t)

        if idx!=0 and idx%config.lrstep==0 and config.lrdecay:
            model.decay_learning_rate(0.1)

def eval(model, config, dataloader):
    accuracy_h5 = []

    #load model

    print("Evaluating the best 5shot model...")
    for i in range(50):
        scores =[]
        for idx, episode in enumerate(dataloader['meta_test']):
            support_data, support_label = episode['train']
            query_data, query_label = episode['test']
            support_data, support_label = support_data.to(config.device), support_label.to(config.device)
            query_data, query_label = query_data.to(config.device), query_label.to(config.device)

            accs = model.evaluate(support_data, support_label)
            accs_ = [cuda.to_cpu(acc) for acc in accs] #이거머지
            score = np.asarray(accs_, dtype=int)
            scores.append(score)
        accuracy_t = 100*np.mean(np.array(scores))
        accuracy_h5.extend([accuracy_t.tolist()])
        print(('600 episodes with 15-query accuracy: 5-shot = {:.2f}%').format(accuracy_t))

        del(accs)
        del(accs_)
        del(accuracy_t)

        #sio.savemat(savefile_name, {'accuracy_h_val':accuracy_h_val, 'accuracy_h_test':accuracy_h_test, 'epoch_best':epoch_best,'acc_best':acc_best, 'accuracy_h5':accuracy_h5})

    print(('Accuracy_test 5 shot ={:.2f}%').format(np.mean(accuracy_h5)))

def main():
    config = parser_args()
    exp_name = experiment_name_generator(config)
    dataloader = data_loader(config)

    model = TapNet(config, dataloader, exp_name)
    model.to(config.device)

    if config.device =='cuda':
        model = nn.DataParallel(model)
        cudnn.benchmark = True

    train(model, config, dataloader, exp_name)


if __name__ == '__main__':
    main()