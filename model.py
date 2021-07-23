import torch
import torch.nn as nn
import torch.nn.functional as F

class TapNet(nn.Module):
    def __init__(self, nb_class_train, nb_class_test, input_size, dimension, n_shot, gpu=1):
        """
        Args
            nb_class_train: number of classes in a training episode
            nb_class_test: number of classes in a test episode
            input_size: dimension of input vector
            dimension:dimension of embedding space
            n_shot: number of shots
        """
        super().__init__()
        self.conv1_1 = nn.Conv2d(None, 64, (3,3), padding=1),
        self.norm1_1 = nn.BatchNorm1d(64),
        self.conv1_2 = nn.Conv2d(64, 64, (3,3), padding=1),
        self.norm1_2 = nn.BatchNorm1d(64),
        self.conv1_3 = nn.Conv2d(64, 64, (3,3), padding=1),
        self.norm1_3 = nn.BatchNorm1d(64),
        self.conv1_r = nn.Conv2d(None, 64, (3,3), padding=1),
        self.norm1_r = nn.BatchNorm1d(64),

        self.conv2_1 = nn.Conv2d(64, 128, (3,3), padding=1),
        self.norm2_1 = nn.BatchNorm1d(128),
        self.conv2_2 = nn.Conv2d(128, 128, (3,3), padding=1),
        self.norm2_2 = nn.BatchNorm1d(128),
        self.conv2_3 = nn.Conv2d(128, 128, (3,3), padding=1),
        self.norm2_3 = nn.BatchNorm1d(128),
        self.conv2_r = nn.Conv2d(64, 128, (3,3), padding=1),
        self.norm2_r = nn.BatchNorm1d(128),

        self.conv3_1 = nn.Conv2d(128, 256, (3,3), padding=1),
        self.norm3_1 = nn.BatchNorm1d(256),
        self.conv3_2 = nn.Conv2d(256, 256, (3,3), padding=1),
        self.norm3_2 = nn.BatchNorm1d(256),
        self.conv3_3 = nn.Conv2d(256, 256, (3,3), padding=1),
        self.norm3_3 = nn.BatchNorm1d(256),
        self.norm3_r = nn.BatchNorm1d(256),

        self.conv4_1 = nn.Conv2d(256, 512, (3,3), padding=1),
        self.norm4_1 = nn.BatchNorm1d(512),
        self.conv4_2 = nn.Conv2d(512, 512, (3,3), padding=1),
        self.norm4_2 = nn.BatchNorm1d(512),
        self.conv4_3 = nn.Conv2d(512,512,(3,3), padding=1),
        self.norm4_3 = nn.BatchNorm1d(512),
        self.conv4_r = nn.Conv2d(256, 512, (3,3), padding=1),
        self.norm4_r = nn.BatchNorm1d(512),

        self.phi = nn.Linear(dimension, nb_class_train)

    def forward(self, x, batchsize, train=True):
        x2 = F.reshape(x, (batchsize, 84, 84, 3))
        x3 = F.transpose(x2, [0,3,1,2])

        c1_r = self.conv1_r(x3)
        n1_r = self.norm1_r(c1_r)

        c1_1 = self.conv1_1(x3)





