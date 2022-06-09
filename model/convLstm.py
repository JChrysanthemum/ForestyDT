
from os.path import join as pj
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from torchsummary import summary
import argparse
import os
import numpy as np
# import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

class GaussianNoise(nn.Module):
    def __init__(self, stddev):
        super().__init__()
        self.stddev = stddev

    def forward(self, din):
        if self.training:
            return din + torch.autograd.Variable(torch.randn(din.size()).cuda() * self.stddev)
        return din

# class CNN64(nn.Module):
#     def __init__(self, rnn_input_size =100):
#         super().__init__()
#         self.conv1 = nn.Conv2d(1, 1, 5)
#         self.conv1_drop = nn.Dropout2d()
#         self.pool = nn.MaxPool2d(4, 4)
#         self.conv2 = nn.Conv2d(1, 1, 5)
#         self.norm = nn.BatchNorm2d(1, 0.8)

#     def forward(self, x):
#         """
#         return:: batch_size, 225
#         """
#         # x = F.relu(self.pool(self.conv1(x)))
#         # x = self.conv1_drop(self.conv1(x))
#         # x = x.view(-1, 150)
#         # x = self.norm(x)
#         x = self.conv1(x)
#         x = self.pool(x)
#         x = self.conv2(x)
#         # x = self.norm(x)
#         x = torch.flatten(x, 1) # flatten all dimensions except batch
#         return x
class CNN64(nn.Module):
    def __init__(self, rnn_input_size =200):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 2, 2)
        self.pool = nn.MaxPool2d(3, 3)
        self.conv2 = nn.Conv2d(2, 4, 3)
        self.fc1 = nn.Linear(144, rnn_input_size)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        # x =  x.view(-1, 200)
        return x

class CNN128(nn.Module):
    def __init__(self, rnn_input_size =200):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 2, 4)
        self.pool = nn.MaxPool2d(4, 4)
        self.conv2 = nn.Conv2d(2, 4, 5)
        self.fc1 = nn.Linear(144, rnn_input_size)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        # x =  x.view(-1, 200)
        return x

class CNN256(nn.Module):
    def __init__(self, rnn_input_size =100):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 3, 4)
        self.pool = nn.MaxPool2d(4, 4)
        self.conv2 = nn.Conv2d(3, 1, 4)

    def forward(self, x):
        """
        return:: batch_size, 225
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        return x


class ConvLstm(nn.Module):
    def __init__(self,img_size=64, latent_dim = 16):
        super(ConvLstm, self).__init__()
        _cnn_dict = {
            64: [CNN64,121],
            128: [CNN128,200],
            256: [CNN128,225]
        }
        cnn_type =  _cnn_dict[img_size][0]
        rnn_input = _cnn_dict[img_size][1]
        self.outsize = latent_dim

        self.cnn = cnn_type(rnn_input)
        self.rnn = nn.LSTM(
            input_size=rnn_input, 
            hidden_size=self.outsize,  # 64
            num_layers=1,
            batch_first=True)
        # self.linear = nn.Linear(64,latent_dim)
        self.norm = nn.BatchNorm1d(self.outsize)
        
        self.rand = GaussianNoise(0.2)
        for param in self.rand.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        # Input : batch_size, timesteps, channels, high, width
        batch_size, timesteps, channels, high, width = x.size()
        c_in = x.view(batch_size * timesteps, channels, high, width)
        c_out = self.cnn(c_in)
        # CNN output : batch_size*timesteps, rnn_input_dim

        # rnn input : batch_size, timesteps, rnn_input_dim
        # print(c_out)
        r_in = c_out.view(batch_size, timesteps, -1)
        r_out, (h_n, h_c) = self.rnn(r_in)
        # rnn input : batch_size, timesteps, hidden_size
        # print(r_out)

        # gen input : batch_size, latent_dim
        out = r_out[:, -1, :]
        out = self.rand(out)

        return out


if __name__ == "__main__":
    pass
            
