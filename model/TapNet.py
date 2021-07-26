import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import cupy as cp

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
        self.criterion = nn.CrossEntropyLoss()
        self.exp_name = exp_name

    def train(self):
        train_loss, accuracy_val = [], []

        acc_best = 0
        epoch_best = 0

        for idx, episode in enumerate(self.dataloader['meta_train']):
            if idx == self.config.n_train_episodes:
                break

            self.optimizer.zero_grad()

            support_data, support_label = episode['train'] # Tensor shape : [1,25,3,84,84]
            query_data, query_label = episode['test'] # Tensor shape : [1,40,3,84,84]
            support_data, support_label = support_data.to(self.device), support_label.to(self.device)
            query_data, query_label = query_data.to(self.device), query_label.to(self.device)

            support_key = self.EmbeddingNet.forward(support_data.squeeze()) # Tensor shape : [25, 512]
            query_key = self.EmbeddingNet.forward(query_data.squeeze()) # Tensor shape : [40, 512]
            average_key = torch.mean(torch.reshape(support_key, (self.n_shot, self.n_class_train, -1)), dim=0) # Tensor shape : [5, 512]

            batchsize_q = len(query_key.data) # 40
            M = self.Projection_Space(average_key, batchsize_q, self.n_class_train).to(self.device) # Tensor shape : [40, 512, 507]

            r_t_temp = torch.matmul(M.permute(0, 2, 1).contiguous(), torch.unsqueeze(query_key, dim=2)) # Tensor shape : [40, 507, 1]
            r_t = torch.matmul(M,r_t_temp).squeeze() # Tensor shape : [40, 507]

            pow_t = self.compute_power(batchsize_q, query_key, M, self.n_class_train)

            target = query_label.squeeze()
            u = 2 * self.EmbeddingNet.phi(r_t) - pow_t
            loss = self.criterion(u, target)
            loss.backward()

            self.optimizer.step()

            # ----------------------------------
            train_loss.append(loss.item())

            if idx % 50 == 0:
                print("Episode: %d, Train Loss: %f " % (idx, loss))
            """
            if idx != 0 and idx % 500 == 0:
                print("Evaluation in Validation data")
                acc = self.evaluate()

                if acc > acc_best:
                    print('save model')
                    self.save_model()
                    acc_best = acc"""

            if idx != 0 and idx % self.config.lr_step == 0 and self.config.lr_decay:
                self.decay_learning_rate(0.1)

    def evaluate(self, validation=True):
        if validation:
            data = 'meta_val'
            n_episodes = self.config.n_val_episodes
        else:
            data = 'meta_test'
            n_episodes = self.config.n_test_episodes
            self.load_model()

        accs = []

        for idx, episode in enumerate(self.dataloader[data]):
            if idx == n_episodes:
                break

            with torch.no_grad():
                support_data, support_label = episode['train']
                query_data, query_label = episode['test']
                support_data, support_label = support_data.to(self.device), support_label.to(self.device)
                query_data, query_label = query_data.to(self.device), query_label.to(self.device)

                support_key = self.EmbeddingNet.forward(support_data.squeeze())  # Tensor shape : [25, 512]
                query_key = self.EmbeddingNet.forward(query_data.squeeze())  # Tensor shape : [40, 512]
                average_key = torch.mean(torch.reshape(support_key, (self.n_shot, self.n_class_test, -1)), dim=0)  # Tensor shape : [5, 512]
                batchsize_q = len(query_key.data)  # 40
                pow_avg = self.compute_power_avg_phi(average_key, train=False)

                phi_ind = [ind for ind in self.select_phi(average_key, pow_avg)]

                M = self.Projection_Space(average_key, batchsize_q, self.n_class_test, train=False, phi_ind=phi_ind).to(self.device)
                r_t_temp = torch.matmul(M.permute(0, 2, 1).contiguous(),torch.unsqueeze(query_key, dim=2))  # Tensor shape : [40, 507, 1]
                r_t = torch.matmul(M, r_t_temp).squeeze()  # Tensor shape : [40, 507]

                pow_t = self.compute_power(batchsize_q, query_key, M, self.n_class_test, train=False, phi_ind=phi_ind)

                #accs_tmp = self.compute_accuracy(query_label, r_t, pow_t, batchsize_q, self.n_class_test, phi_ind=phi_ind)
                #accs.append(accs_tmp)

        return accs

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
        M = torch.tensor(M).expand(batchsize, self.dim, self.dim-n_class)
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

    def compute_accuracy(self, t_data, r_t, pow_t, batchsize, n_class, phi_ind=None):
        ro = 2*self.EmbeddingNet.phi(r_t)
        ro_t = torch.cuda.FloatTensor(ro.data[:, phi_ind])
        u = ro_t-pow_t
        t_est = cp.argmax(nn.Softmax(u).data)

        return t_est == cp.array(t_data)

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
                    k= k-1
                phi_ind[i] = np.int(u_avg_ind[i,k])
        return phi_ind.tolist()

    def save_model(self):
        torch.save(self.EmbeddingNet.state_dict(),)
        torch.save(self.Embedding_module.state_dict(),
                   f'{self.config.modelweights}{self.config.dataset}/{self.EXP_NAME}/{str(save_num).zfill(6)}_embedder.pkl')

        return NotImplemented

    def load_model(self):
        """
           if save_num =='Latest':
            # Get the latest saved file
            files = os.listdir(f"{self.config.modelweights}{self.config.dataset}/{self.EXP_NAME}")
            files.sort()
            save_num = files[-1].split('_')[0]
        self.Embedding_module.load_state_dict(torch.load(f'{self.config.modelweights}{self.config.dataset}/{self.EXP_NAME}/{save_num}_embedder.pkl'),map_location=self.device)
        self.Local_classifier.load_state_dict(torch.load(f'{self.config.modelweights}{self.config.dataset}/{self.EXP_NAME}/{save_num}_classifier.pkl'),map_location=self.device)


        :return:
        """
        return NotImplemented

    def decay_learning_rate(self, decaying_parameter=0.5):
        self.optimizer.lr = self.optimizer.lr * decaying_parameter



