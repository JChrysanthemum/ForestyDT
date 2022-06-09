import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
import numpy as np
from dataWrapper import SampleInput

class VariationalEncoder(nn.Module):
    def __init__(self, latent_dim, img_size=128,in_seq_len=4):
        super(VariationalEncoder, self).__init__()
        stride=4
        self.conv1 = nn.Conv2d(1, 1, 5, stride=stride)
        conved_size = (img_size//stride -1 + (img_size%stride != 0)*1)**2
        self.linear1 = nn.Linear(in_seq_len*conved_size, 512)
        self.linear2 = nn.Linear(512, latent_dim)
        self.linear3 = nn.Linear(512, latent_dim)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x):
        batch_size, timesteps, channels, high, width = x.size()
        c_in = x.view(batch_size * timesteps, channels, high, width)
        c_out = self.conv1(c_in)
        
        l_in = c_out.view(batch_size , -1)

        # print(c_in.size())
        # print(c_out.size())
        # print(l_in.size())
        # exit()

        # l_in = torch.flatten(c_out, start_dim=1)
        
        l_out = F.relu(self.linear1(l_in))
        mu =  self.linear2(l_out)
        sigma = torch.exp(self.linear3(l_out))
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z
 
class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 512)
        self.linear2 = nn.Linear(512, 784)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))
        return z.reshape((-1, 1, 28, 28))   
    
class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
    
if __name__ == "__main__":
    img_seq = SampleInput().img_seq
    model = VariationalAutoencoder(200)
    print(img_seq.size())
    model(img_seq)