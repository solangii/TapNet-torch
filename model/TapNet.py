import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
#from torch.utils.tensorboard import SummaryWriter

import numpy as np
import cupy as cp

from model.model_ResNet12 import EmbeddingNet
from model.utils import nullspace_torch


class TapNet:
    def __init__(self, config, dataloader, exp_name):
        self.config = config
        self.dataloader = dataloader

        self.device = config.device

        self.n_shot = config.n_shot
        self.n_class_train = config.n_class_train
        self.n_class_test = config.n_class_test
        self.dim = config.dim

        self.EmbeddingNet = EmbeddingNet(self.dim, self.n_class_train).to(self.device)

        if config.use_parallel:
            self.EmbeddingNet = nn.DataParallel(self.EmbeddingNet)
            cudnn.benchmark = True

        self.optimizer = optim.Adam(list(self.EmbeddingNet.parameters()), lr=self.config.lr) #Todo wd 넣
        self.criterion = nn.CrossEntropyLoss()

        self.exp_name = exp_name

        #self.writer = SummaryWriter('tensorboard/')

    def train(self):
        train_loss, accuracy_val = [], []

        acc_best = 0

        for idx, episode in enumerate(self.dataloader['meta_train']):
            if idx == self.config.n_train_episodes:
                break

            self.optimizer.zero_grad()

            support_data, support_label = episode['train'] # Tensor shape : [1,25,3,84,84]
            query_data, query_label = episode['test'] # Tensor shape : [1,40,3,84,84]
            support_data, support_label = support_data.to(self.device), support_label.to(self.device)
            query_data, query_label = query_data.to(self.device), query_label.to(self.device)
            self.EmbeddingNet.train()

            support_data.requires_grad = True
            query_data.requires_grad = True

            support_key = self.EmbeddingNet.forward(support_data.squeeze()) # Tensor shape : [25, 512]
            query_key = self.EmbeddingNet.forward(query_data.squeeze()) # Tensor shape : [40, 512]
            average_key = torch.mean(torch.reshape(support_key, (self.n_shot, self.n_class_train, -1)), dim=0) # Tensor shape : [5, 512]

            M = self.projection_space(average_key, self.n_class_train)
            query_M = torch.matmul(query_key, M)
            classifier_M = torch.matmul(self.EmbeddingNet.phi.weight, M)

            loss = self.compute_loss(query_label, query_M, classifier_M)
            train_loss.append(loss.item())
            loss.backward()
            self.optimizer.step()

            # ----------------------------------

            if idx % 50 == 0:
                print("Episode: %d, Train Loss: %f " % (idx, loss))
                #self.writer.add_scalar('training loss', loss, idx)

            if idx != 0 and idx % 500 == 0:
                print("Evaluation in Validation data")
                acc = self.evaluate()

                if acc > acc_best:
                    print('save model')
                    self.save_model(acc, idx)

                    acc_best = acc
                    epi_best = idx
                accuracy_val.extend([acc.tolist()])
                print("Validation accuracy : %f " % (acc))


            #if idx != 0 and idx % self.config.lr_step == 0 and self.config.lr_decay:
            #    self.decay_learning_rate(0.1)

    def evaluate(self, validation=True, PATH=None):
        if validation:
            data = 'meta_val'
            n_episodes = self.config.n_val_episodes
        else:
            data = 'meta_test'
            n_episodes = self.config.n_test_episodes
            self.load_model(PATH)

        accs = []

        with torch.no_grad():
            for idx, episode in enumerate(self.dataloader[data]):
                if idx == n_episodes:
                    break

                support_data, support_label = episode['train']
                query_data, query_label = episode['test']
                support_data, support_label = support_data.to(self.device), support_label.to(self.device)
                query_data, query_label = query_data.to(self.device), query_label.to(self.device)
                self.EmbeddingNet.eval()

                support_key = self.EmbeddingNet.forward(support_data.squeeze())  # Tensor shape : [25, 512]
                query_key = self.EmbeddingNet.forward(query_data.squeeze())  # Tensor shape : [40, 512]
                average_key = torch.mean(torch.reshape(support_key, (self.n_shot, self.n_class_test, -1)), dim=0)  # Tensor shape : [5, 512]

                pow_avg = self.compute_power_avg_phi(average_key, train=False)
                phi_ind = [np.int(ind) for ind in self.select_phi(average_key, pow_avg)]

                M = self.projection_space(average_key, self.n_class_test, train=False, phi_ind=phi_ind)
                query_M = torch.matmul(query_key, M)
                classifier_M = torch.matmul(self.EmbeddingNet.phi.weight, M)

                accs_tmp = self.compute_accuracy(query_label, query_M, classifier_M, phi_ind=phi_ind)
                accs.append(accs_tmp)

        accs = torch.cuda.FloatTensor(accs)
        acc = torch.mean(accs)
        return acc

    def projection_space(self, average_key, n_class, train=True, phi_ind=None):
        c_t = average_key
        eps = 1e-6

        if train:
            Phi_tmp = self.EmbeddingNet.phi.weight
        else:
            Phi_data = self.EmbeddingNet.phi.weight.data
            Phi_tmp = torch.cuda.FloatTensor(Phi_data[phi_ind, :])

        Phi_sum = Phi_tmp.sum(dim=0)
        Phi = n_class * Phi_tmp - Phi_sum

        power_Phi = torch.sqrt(torch.sum(Phi * Phi, 1))
        power_Phi = torch.t(power_Phi.expand(self.dim, n_class))
        Phi = Phi / (power_Phi + eps)

        power_c = torch.sqrt(torch.sum(c_t * c_t, 1))
        power_c = torch.t(power_c.expand(self.dim, n_class))
        c_tmp = c_t / (power_c + eps)

        null = Phi - c_tmp
        M = nullspace_torch(null)
        return M

    def compute_power(self, batchsize, key, M, n_class, train=False, phi_ind=None):
        if train:
            Phi_out = self.EmbeddingNet.phi.weight
        else:
            Phi_data = self.EmbeddingNet.phi.weight.data
            Phi_out = torch.cuda.FloatTensor(Phi_data[phi_ind, :])  # Tensor shape : [1, 5, 512]

        Phi_out_batch = Phi_out.expand(batchsize, n_class, self.dim) # Tensor shape : [40, 5, 512]
        PhiM = torch.matmul(Phi_out_batch, M)  # Tensor shape : [40, 512, 507]
        PhiMs = torch.sum(PhiM*PhiM, dim=2)  # Tensor shape : [40, 5]

        key_t = torch.reshape(key, (batchsize, 1, self.dim))  # Tensor shape : [40, 1, 512]
        keyM = torch.matmul(key_t, M)  # Tensor shape : [40, 1, 507]
        keyMs = torch.sum(keyM*keyM, dim=2).expand(batchsize, n_class)  # Tensor shape : [40, 5]

        pow_t = PhiMs + keyMs

        return pow_t

    def compute_power_avg_phi(self, average_key, train=False):
        avg_pow = torch.sum(average_key*average_key, dim=1)
        Phi = self.EmbeddingNet.phi.weight
        Phis = torch.sum(Phi*Phi, dim=1)

        avg_pow_bd = torch.reshape(avg_pow, (len(avg_pow),1)).expand(len(avg_pow), len(Phis))
        wzs_bd = torch.reshape(Phis, (1, len(Phis))).expand(len(avg_pow), len(Phis))

        pow_avg = avg_pow_bd + wzs_bd

        return pow_avg

    def select_phi(self, average_key, avg_pow):
        u_avg = 2*self.EmbeddingNet.phi(average_key).data
        u_avg = u_avg - avg_pow.data
        u_avg_ind = cp.asnumpy(cp.argsort(u_avg, axis=1))
        phi_ind = np.zeros(self.n_class_test)

        for i in range(self.n_class_test):
            if i == 0:
                phi_ind[i] = np.int(u_avg_ind[i, self.n_class_train-1])
            else:
                k = self.n_class_train-1
                while u_avg_ind[i,k] in phi_ind[:i]:
                    k = k-1
                phi_ind[i] = np.int(u_avg_ind[i,k])
        return phi_ind.tolist()

    def compute_euclidean(self, keys, weight):
        """Computes the logits of being in one cluster, squared Euclidean.
                Args:
                weight: [K, D] Cluster center representation.
                keys: [N, D] Data representation.
                """
        weight = weight.unsqueeze(0)
        keys = keys.unsqueeze(1)
        neg_dist = -torch.sum(torch.square(keys - weight), dim=-1)
        return neg_dist

    def compute_loss(self, label, query, classifier):
        labels = label.squeeze()
        logits = self.compute_euclidean(query, classifier)
        return self.criterion(logits, labels)

    def compute_accuracy(self, t_data, query, classifier, phi_ind=None):
        labels = t_data.squeeze()
        logits = self.compute_euclidean(query, classifier)[:, phi_ind]
        t_est = torch.argmax(F.softmax(logits), dim=1)

        result = (t_est == labels)
        acc = torch.sum(result == True) / labels.shape[0]
        return acc

    def save_model(self, acc, idx):
        state = {
            'net': self.EmbeddingNet.state_dict(),
            'acc': acc,
            'idx': idx,
        }
        if not os.path.isdir(self.config.save_root):
            os.mkdir(self.config.save_root)

        name = self.config.save_root + self.exp_name + '.pth'
        torch.save(state, name)

    def load_model(self, path):
        self.EmbeddingNet = torch.load(path)

"""
    def decay_learning_rate(self, decaying_parameter=0.5):
        self.optimizer.weight = self.optimizer.lr * decaying_parameter"""





