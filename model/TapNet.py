import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

from model.model_ResNet12 import EmbeddingNet
from model.utils import nullspace_gpu


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

        self.optimizer = optim.Adam(params=list(self.EmbeddingNet.parameters()), lr=self.config.lr) #wd 는 어디에?
        #self.creterion = self.compute_loss() #이거ㅓ 맞는지 모르겠
        self.exp_name = exp_name

    def train(self):
        train_loss, accuracy_val = [], []

        acc_best = 0
        epoch_best = 0

        for idx, episode in enumerate(self.dataloader['meta_train']):

            self.optimizer.zero_grad()

            support_data, support_label = episode['train'] # Tensor shape : [1,25,3,84,84]
            query_data, query_label = episode['test'] # Tensor shape : [1,40,3,84,84]
            support_data, support_label = support_data.to(self.device), support_label.to(self.device)
            query_data, query_label = query_data.to(self.device), query_label.to(self.device)

            support_key = self.EmbeddingNet.forward(support_data.squeeze()) # Tensor shape : [25, 2048]
            query_key = self.EmbeddingNet.forward(query_data.squeeze()) # Tensor shape : [40, 2048]
            average_key = torch.mean(torch.reshape(support_key, (self.n_shot, self.n_class_train, -1)), dim=0)

            batchsize_q = len(query_key.data) # 40
            M = self.Projection_Space(average_key, batchsize_q, self.n_class_train)

            r_t = torch.reshape(torch.matmul(M, torch.matmul(torch.t(M), query_key)), (batchsize_q, -1))
            pow_t = self.compute_power(batchsize_q, query_key, M, self.n_class_train)

            loss = self.compute_loss(query_label, r_t, pow_t, batchsize_q, self.n_class_train)

            loss.backward()
            self.optimizer.step()

            # ----------------------------------
            train_loss += loss.item()

            if idx % 50 == 0:
                print("Episode: %d, Train Loss: %f " % (idx, loss))

            if idx != 0 and idx % 500 == 0:
                print("Evaluation in Validation data")
                acc = self.evaluate()

                if acc > acc_best:
                    print('save model')
                    self.save_model()
                    acc_best = acc

            if idx != 0 and idx % self.config.lrstep == 0 and self.config.lrdecay:
                self.decay_learning_rate(0.1)


        return None

    def evaluate(self, val_mode=True):
       return NotImplemented

    def Projection_Space(self, average_key, batchsize, n_class, train=True, phi_ind=None):
        c_t = average_key
        eps=1e-6

        if train==True:
            Phi_tmp = self.EmbeddingNet.phi.weight
        else:
            Phi_data = self.EmbeddingNet.phi.weight.data
            Phi_tmp = torch.cuda.FloatTensor(Phi_data[phi_ind,:])

        Phi_sum = Phi_tmp.sum(dim=0)
        Phi = n_class*Phi_tmp - Phi_sum

        power_Phi = torch.sqrt(torch.sum(Phi*Phi, 1))
        power_Phi = torch.t(power_Phi.expand(self.dim, n_class))
        Phi = Phi / (power_Phi + eps)

        power_c = torch.sqrt(torch.sum(c_t * c_t, 1))
        power_c = torch.t(power_c.expand(self.dim, n_class))
        c_tmp = c_t/(power_c + eps)

        null = Phi - c_tmp
        M = nullspace_gpu(null.data)
        M = torch.reshape(M, (batchsize, self.dim, self.dim-n_class))
        return M

    def decay_learning_rate(self, decaying_parameter = 0.5):
        self.optimizer.lr = self.optimizer.lr * decaying_parameter
        return NotImplemented

    def compute_power(self, batchsize, key, M, n_class, train=False, phi_ind=None):
        if train:
            Phi_out = self.EmbeddingNet.phi.weight
        else:
            Phi_data = self.EmbeddingNet.phi.weight.data
            Phi_out = torch.cuda.FloatTensor(Phi_data[phi_ind, :])
        Phi_out_batch = torch.reshape(Phi_out, (batchsize, n_class, self.dim))
        PhiM = torch.matmul(Phi_out_batch, M)
        PhiMs = torch.sum(PhiM*PhiM, dim=2)

        key_t = torch.reshape(key, (batchsize, 1, self.dim))
        keyM = torch.matmul(key_t, M)
        keyMs = torch.sum(keyM*keyM, dim=2)

        pow_t = PhiMs + keyMs

        return pow_t

    def compute_power_avg_phi(self, batchsize, n_class, average_key, train=False):
        avg_pow = torch.sum(average_key*average_key, dim=1)
        Phi = self.EmbeddingNet.phi.weight
        Phis = torch.sum(Phi*Phi, dim=1)

        avg_pow_bd = torch.reshape(avg_pow, (len(avg_pow),1)).expand(len(avg_pow), len(Phis))
        wzs_bd = torch.reshape(Phis, (1, len(Phis))).expand(len(avg_pow), len(Phis))

        pow_avg = avg_pow_bd + wzs_bd

        return pow_avg

    def compute_loss(self, t_data, r_t, pow_t, batchsize, n_class, train=True):
        t = torch.cuda.FloatTensor(np.array(t_data, dtype=np.int32))
        u = 2*self.EmbeddingNet.phi(r_t) - pow_t
        loss = nn.CrossEntropyLoss(u,t)
        return loss

    def compute_accuracy(self):
        return None

    def select_phi(self, average_key, avg_pow):
        return NotImplemented

    def save_model(self):
        return NotImplemented


