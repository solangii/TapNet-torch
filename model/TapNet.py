import torch
import torch.nn as nn
import torch.optim as optim

from model.model_ResNet12 import EmbeddingNet
from model.utils import nullspace

class TapNet:
    def __init__(self, config, dataloader, exp_name):
        self.config = config
        self.dataloader = dataloader

        self.device = config.device

        self.n_shot = config.n_shot
        self.n_class_train = config.n_class_train
        self.n_class_test = config.n_class_test
        self.dim = config.dim

        self.EmbeddingNet = EmbeddingNet(self.dim, self.n_class_train).to(self.device) #파라미터 왜필?

        self.optimizer = optim.Adam(params=list(self.EmbeddingNet.parameters()), lr=self.config.lr) #wd 는 어디에?
        self.creterion = self.compute_loss()
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

            support_key = self.EmbeddingNet.forward(support_data) # Tensor shape : [25, 2048]
            query_key = self.EmbeddingNet.forward(query_data) # Tensor shape : [40, 2048]
            support_key = support_key.view(self.n_shot, self.n_class_train, -1) # Tensor shape : [5,5,2048]
            average_key = support_key.mean()

            batchsize_q = len(query_key.data) # 40
            M = self.Projection_Space(average_key, batchsize_q, self.n_class_train)


            #r_t

            pow_t = self.compute_power()

            loss = self.compute_loss()

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

        for i in range(n_class):
            if i == 0:
                Phi_sum = Phi_tmp[i] # shape : 512
            else:
                Phi_sum += Phi_tmp[i]

        Phi = n_class*Phi_tmp - Phi_tmp.repeat(n_class) # 내 예상 phi는

        power_Phi = torch.sqrt(torch.sum(c_t * c_t))
        power_Phi = power_Phi.view(self.dim, n_class)
        #power_Phi = F.transpose(F.broadcast_to(power_Phi, [self.dimension, nb_class]))

        Phi = Phi / (power_Phi + eps)

        power_c = F.sqrt(F.sum(c_t * c_t, axis=1))
        power_c = F.transpose(F.broadcast_to(power_c, [self.dimension, nb_class]))
        c_tmp = c_t/(power_c + eps)

        null = Phi - c_tmp
        M = nullspace(null.data)
        # M = F.broadcast_to(M,[batchsize, self.dimension, self.dimension-nb_class])
        # M : N, 512, 492 사이즈로 변경
        return M

    def decay_learning_rate(self, decaying_parameter = 0.5):
        self.optimizer.lr = self.optimizer.lr * decaying_parameter
        return NotImplemented

    def compute_power(self, batchsize, n_class, average_key, train=False):
        avg_pow = average_key*average_key
        return NotImplemented

    def compute_power_avg_phi(self):
        return NotImplemented

    def compute_loss(self):
        return NotImplemented

    def compute_accuracy(self):
        return None

    def select_phi(self):
        return NotImplemented

    def save_model(self):
        return NotImplemented


