import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingNet(nn.Module):
    def __init__(self, dim, n_class_train):
        super(EmbeddingNet, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, bias=False),
                                   nn.BatchNorm2d(num_features=64),
                                   nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
                                   nn.BatchNorm2d(num_features=64),
                                   nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
                                   nn.BatchNorm2d(num_features=64))
        self.conv1_r = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, bias=False),
                                     nn.BatchNorm2d(num_features=64))
        self.pool1 = nn.Sequential(nn.MaxPool2d(kernel_size=2),
                                   nn.Dropout2d(p=0.3))

        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, bias=False),
                                   nn.BatchNorm2d(128),
                                   nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=False),
                                   nn.BatchNorm2d(128),
                                   nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=False),
                                   nn.BatchNorm2d(128))
        self.conv2_r = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, bias=False),
                                     nn.BatchNorm2d(128))
        self.pool2 = nn.Sequential(nn.MaxPool2d(kernel_size=2),
                                   nn.Dropout2d(p=0.2))

        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, bias=False),
                                   nn.BatchNorm2d(256),
                                   nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False),
                                   nn.BatchNorm2d(256),
                                   nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False),
                                   nn.BatchNorm2d(256))
        self.conv3_r = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, bias=False),
                                     nn.BatchNorm2d(256))
        self.pool3 = nn.Sequential(nn.MaxPool2d(kernel_size=2),
                                   nn.Dropout2d(p=0.2))

        self.conv4 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, bias=False),
                                   nn.BatchNorm2d(512),
                                   nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, bias=False),
                                   nn.BatchNorm2d(512),
                                   nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, bias=False),
                                   nn.BatchNorm2d(512))
        self.conv4_r = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, bias=False),
                                     nn.BatchNorm2d(512))
        self.pool4 = nn.Sequential(nn.MaxPool2d(kernel_size=2),
                                   nn.Dropout2d(p=0.2),
                                   nn.AvgPool2d(kernel_size=2)) # 6dmfh

        self.phi = nn.Linear(dim, n_class_train)

    def forward(self, x):
        x = x.view(-1, 84, 84, 3)
        x = x.permute(0, 3, 1, 2).contiguous()  # shape = (25, 3, 84, 84)

        out1 = self.pool1(F.relu(self.conv1(x) + self.conv1_r(x)))
        out2 = self.pool2(F.relu(self.conv2(out1) + self.conv2_r(out1)))
        out3 = self.pool3(F.relu(self.conv3(out2) + self.conv3_r(out2)))
        out4 = self.pool4(F.relu(self.conv4(out3) + self.conv4_r(out3)))
        h_t = out4.view(x.shape[0], -1)

        return h_t
