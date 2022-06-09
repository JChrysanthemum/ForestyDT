import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class WGAN_Generator(nn.Module):
    def __init__(self, img_size=64, latent_dim=100,channels=1):
        super().__init__()
        self.img_shape = (channels,img_size, img_size)
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *self.img_shape)
        return img

class DeConv(nn.Module):
    def __init__(self,img_size=256,latent_dim=200,channels=3):
        super().__init__()
        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class CGAN_Generator(nn.Module):
    def __init__(self, img_size=64, latent_dim=100,channels=1, n_classes=10, out_dim = -1):
        super().__init__()
        if out_dim ==-1:
            out_dim = latent_dim
        self.label_emb = nn.Embedding(n_classes, latent_dim)
        self.img_shape = (channels,img_size, img_size)
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim + latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, z, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((self.label_emb(labels), z), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *self.img_shape)
        return img

class CONWGAN_Generator(nn.Module):
    class MyConvo2d(nn.Module):
        def __init__(self, input_dim, output_dim, kernel_size, he_init = True,  stride = 1, bias = True):
            super(CONWGAN_Generator.MyConvo2d, self).__init__()
            self.he_init = he_init
            self.padding = int((kernel_size - 1)/2)
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride=1, padding=self.padding, bias = bias)

        def forward(self, input):
            output = self.conv(input)
            return output

    class ConvMeanPool(nn.Module):
        def __init__(self, input_dim, output_dim, kernel_size, he_init = True):
            super(CONWGAN_Generator.ConvMeanPool, self).__init__()
            self.he_init = he_init
            self.conv = CONWGAN_Generator.MyConvo2d(input_dim, output_dim, kernel_size, he_init = self.he_init)

        def forward(self, input):
            output = self.conv(input)
            output = (output[:,:,::2,::2] + output[:,:,1::2,::2] + output[:,:,::2,1::2] + output[:,:,1::2,1::2]) / 4
            return output

    class MeanPoolConv(nn.Module):
        def __init__(self, input_dim, output_dim, kernel_size, he_init = True):
            super(CONWGAN_Generator.MeanPoolConv, self).__init__()
            self.he_init = he_init
            self.conv = CONWGAN_Generator.MyConvo2d(input_dim, output_dim, kernel_size, he_init = self.he_init)

        def forward(self, input):
            output = input
            output = (output[:,:,::2,::2] + output[:,:,1::2,::2] + output[:,:,::2,1::2] + output[:,:,1::2,1::2]) / 4
            output = self.conv(output)
            return output

    class DepthToSpace(nn.Module):
        def __init__(self, block_size):
            super(CONWGAN_Generator.DepthToSpace, self).__init__()
            self.block_size = block_size
            self.block_size_sq = block_size*block_size

        def forward(self, input):
            output = input.permute(0, 2, 3, 1)
            (batch_size, input_height, input_width, input_depth) = output.size()
            output_depth = int(input_depth / self.block_size_sq)
            output_width = int(input_width * self.block_size)
            output_height = int(input_height * self.block_size)
            t_1 = output.reshape(batch_size, input_height, input_width, self.block_size_sq, output_depth)
            spl = t_1.split(self.block_size, 3)
            stacks = [t_t.reshape(batch_size,input_height,output_width,output_depth) for t_t in spl]
            output = torch.stack(stacks,0).transpose(0,1).permute(0,2,1,3,4).reshape(batch_size,output_height,output_width,output_depth)
            output = output.permute(0, 3, 1, 2)
            return output

    class UpSampleConv(nn.Module):
        def __init__(self, input_dim, output_dim, kernel_size, he_init = True, bias=True):
            super(CONWGAN_Generator.UpSampleConv, self).__init__()
            self.he_init = he_init
            self.conv = CONWGAN_Generator.MyConvo2d(input_dim, output_dim, kernel_size, he_init = self.he_init, bias=bias)
            self.depth_to_space = CONWGAN_Generator.DepthToSpace(2)

        def forward(self, input):
            output = input
            output = torch.cat((output, output, output, output), 1)
            output = self.depth_to_space(output)
            output = self.conv(output)
            return output

    class ResidualBlock(nn.Module):
        def __init__(self, input_dim, output_dim, kernel_size, hw, resample=None):
            super(CONWGAN_Generator.ResidualBlock, self).__init__()

            self.input_dim = input_dim
            self.output_dim = output_dim
            self.kernel_size = kernel_size
            self.resample = resample
            self.bn1 = None
            self.bn2 = None
            self.relu1 = nn.ReLU()
            self.relu2 = nn.ReLU()
            if resample == 'down':
                self.bn1 = nn.LayerNorm([input_dim, hw, hw])
                self.bn2 = nn.LayerNorm([input_dim, hw, hw])
            elif resample == 'up':
                self.bn1 = nn.BatchNorm2d(input_dim)
                self.bn2 = nn.BatchNorm2d(output_dim)
            elif resample == None:
                #TODO: ????
                self.bn1 = nn.BatchNorm2d(output_dim)
                self.bn2 = nn.LayerNorm([input_dim, hw, hw])
            else:
                raise Exception('invalid resample value')

            if resample == 'down':
                self.conv_shortcut = CONWGAN_Generator.MeanPoolConv(input_dim, output_dim, kernel_size = 1, he_init = False)
                self.conv_1 = CONWGAN_Generator.MyConvo2d(input_dim, input_dim, kernel_size = kernel_size, bias = False)
                self.conv_2 = CONWGAN_Generator.ConvMeanPool(input_dim, output_dim, kernel_size = kernel_size)
            elif resample == 'up':
                self.conv_shortcut = CONWGAN_Generator.UpSampleConv(input_dim, output_dim, kernel_size = 1, he_init = False)
                self.conv_1 = CONWGAN_Generator.UpSampleConv(input_dim, output_dim, kernel_size = kernel_size, bias = False)
                self.conv_2 = CONWGAN_Generator.MyConvo2d(output_dim, output_dim, kernel_size = kernel_size)
            elif resample == None:
                self.conv_shortcut = CONWGAN_Generator.MyConvo2d(input_dim, output_dim, kernel_size = 1, he_init = False)
                self.conv_1 = CONWGAN_Generator.MyConvo2d(input_dim, input_dim, kernel_size = kernel_size, bias = False)
                self.conv_2 = CONWGAN_Generator.MyConvo2d(input_dim, output_dim, kernel_size = kernel_size)
            else:
                raise Exception('invalid resample value')

        def forward(self, input):
            if self.input_dim == self.output_dim and self.resample == None:
                shortcut = input
            else:
                shortcut = self.conv_shortcut(input)

            output = input
            output = self.bn1(output)
            output = self.relu1(output)
            output = self.conv_1(output)
            output = self.bn2(output)
            output = self.relu2(output)
            output = self.conv_2(output)

            return shortcut + output

    class ReLULayer(nn.Module):
        def __init__(self, n_in, n_out):
            super(CONWGAN_Generator.ReLULayer, self).__init__()
            self.n_in = n_in
            self.n_out = n_out
            self.linear = nn.Linear(n_in, n_out)
            self.relu = nn.ReLU()

        def forward(self, input):
            output = self.linear(input)
            output = self.relu(output)
            return output

    class FCGenerator(nn.Module):
        def __init__(self, OUT_DIM, FC_DIM=512):
            super(CONWGAN_Generator.FCGenerator, self).__init__()
            self.relulayer1 = CONWGAN_Generator.ReLULayer(128, FC_DIM)
            self.relulayer2 = CONWGAN_Generator.ReLULayer(FC_DIM, FC_DIM)
            self.relulayer3 = CONWGAN_Generator.ReLULayer(FC_DIM, FC_DIM)
            self.relulayer4 = CONWGAN_Generator.ReLULayer(FC_DIM, FC_DIM)
            self.linear = nn.Linear(FC_DIM, OUT_DIM)
            self.tanh = nn.Tanh()

        def forward(self, input):
            output = self.relulayer1(input)
            output = self.relulayer2(output)
            output = self.relulayer3(output)
            output = self.relulayer4(output)
            output = self.linear(output)
            output = self.tanh(output)
            return output
    
    def __init__(self, img_size=64, channels=1):
        super(CONWGAN_Generator, self).__init__()

        self.dim = img_size
        self.out_dim = img_size*img_size*channels

        self.ln1 = nn.Linear(128, 4*4*8*self.dim)
        self.rb1 = CONWGAN_Generator.ResidualBlock(8*self.dim, 8*self.dim, 3, self.out_dim, resample = 'up')
        self.rb2 = CONWGAN_Generator.ResidualBlock(8*self.dim, 4*self.dim, 3, self.out_dim, resample = 'up')
        self.rb3 = CONWGAN_Generator.ResidualBlock(4*self.dim, 2*self.dim, 3, self.out_dim, resample = 'up')
        self.rb4 = CONWGAN_Generator.ResidualBlock(2*self.dim, 1*self.dim, 3, self.out_dim, resample = 'up')
        self.bn  = nn.BatchNorm2d(self.dim)

        self.conv1 = CONWGAN_Generator.MyConvo2d(1*self.dim, 3, 3)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, input):
        output = self.ln1(input.contiguous())
        output = output.view(-1, 8*self.dim, 4, 4)
        output = self.rb1(output)
        output = self.rb2(output)
        output = self.rb3(output)
        output = self.rb4(output)

        output = self.bn(output)
        output = self.relu(output)
        output = self.conv1(output)
        output = self.tanh(output)
        output = output.view(-1, self.out_dim)
        return output

