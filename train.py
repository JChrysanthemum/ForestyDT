from operator import le
import os
import time
from os.path import join as pj
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import shutil
from torch.autograd import Variable
from torch.utils import data
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
import sys, os
# from model.convLstm import ConvLstm
from model import Generator,Discriminator, wganLstm
from data import load_ds,QMNP_Seq_L

exit()


def train(img_size = 64 ,channels = 1,n_epochs =10001, batch_size = 8, latent_dim = 100 , embed_out = 10,sample_interval=200, cuda_id=2, band="B3"):
    lr = 0.003
    n_critic = 5
    clip_value = 0.001
    
    _root_path = os.path.abspath(os.path.dirname(__file__))
    data_head="%s_%d_%d_%d_%d"%(band,img_size,batch_size,latent_dim,embed_out)

    img_fd = pj(_root_path,"images", data_head)
    cp_fd = pj(_root_path,"checkpoints", data_head)
    if os.path.exists(img_fd):
        shutil.rmtree(img_fd)
    if os.path.exists(cp_fd):
        shutil.rmtree(cp_fd)

    os.makedirs(img_fd, exist_ok=True)
    os.makedirs(cp_fd, exist_ok=True)

    torch.cuda.set_device(cuda_id)
    
    cuda = True if torch.cuda.is_available() else False
    # cuda = False
    print("Use gpu: ",cuda, cuda_id)

    

    # Configure data loader)

    dataset = load_ds(img_size=img_size,band=band)
    # dataset = ImgDS_Seq(root_path="/home/jiangxt18/Data/FD", 
    #     img_size=img_size,channels=1, seq_len=3)
    fs = dataset.features
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    # Initialize generator and discriminator
    generator = Generator(img_size, latent_dim, channels, embed_in=fs, embed_out=embed_out)
    discriminator = Discriminator(img_size, channels, embed_in=fs, embed_out=embed_out)

    sample_interval *=len(dataloader)

    
    gan_loss1 = torch.nn.MSELoss()

    # Optimizers
    optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=lr)
    optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=lr)

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    
    if cuda:
        generator.cuda()
        discriminator.cuda()

    # ----------
    #  Training
    # ----------

    batches_done = 0
    
    for epoch in range(n_epochs):
        for i, (img_seq, next_img, block_id) in enumerate(dataloader):
            if len(img_seq) <= 1:
                continue
            if cuda:
                img_seq, next_img, block_id = img_seq.cuda(), next_img.cuda(), block_id.type(torch.LongTensor).cuda()
                gan_loss1.cuda()
            img_seq, next_img, block_id= Variable(img_seq), Variable(next_img), Variable(block_id)

            # Configure input
            real_imgs = Variable(next_img.type(Tensor))

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Sample noise as generator input
            # z = Variable(Tensor(np.random.normal(0, 1, (next_img.shape[0], latent_dim))))

            # Generate a batch of images
            fake_imgs = generator(img_seq,block_id).detach()
            # Adversarial loss
            loss_D = -torch.mean(discriminator(real_imgs,block_id)) + torch.mean(discriminator(fake_imgs,block_id))

            loss_D.backward()
            optimizer_D.step()

            # Clip weights of discriminator
            for p in discriminator.parameters():
                p.data.clamp_(-clip_value, clip_value)

            # Train the generator every n_critic iterations
            if i % n_critic == 0:

                # -----------------
                #  Train Generator
                # -----------------

                optimizer_G.zero_grad()

                # Generate a batch of images
                gen_imgs = generator(img_seq,block_id)
                # Adversarial loss
                # loss_G = -torch.mean(discriminator(gen_imgs))
                loss_G = -torch.mean(discriminator(gen_imgs,block_id)) # + gan_loss1(gen_imgs,next_img) 
                # + 0.2*gan_loss2(gen_imgs.detach(),block_id.cuda())

                # print("*"*10)
                # print(torch.mean(discriminator(gen_imgs)))
                # print(gan_loss1(gen_imgs,next_img), 0.7* gan_loss1(gen_imgs,next_img))
                # print(gan_loss2(gen_imgs.detach().cpu(),block_id))
                # print("-"*10)
                # time.sleep(1)
                

                loss_G.backward()
                optimizer_G.step()

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, n_epochs, batches_done % len(dataloader), len(dataloader), loss_D.item(), loss_G.item())
                )

            if batches_done % sample_interval == 0:
                #print("Saving checkpoint and image, plz waiting")
                torch.save(generator, "%s/%d.pt" % (cp_fd, epoch))
                torch.save(generator, "%s/%d-D.pt" % (cp_fd, epoch))
                iter_max = batch_size if batch_size <= len(gen_imgs.data) else len(gen_imgs.data)
                out_imgs = torch.Tensor(np.zeros((2*iter_max, channels, img_size, img_size))).cuda()

                out_list = []
                for i in range(iter_max):
                    out_list.append(img_seq[i].view(dataset.seq_len, 1, img_size, img_size))
                    out_list.append(next_img[i].view(1, 1, img_size, img_size).cuda())
                    out_list.append(gen_imgs.data[i].view(1, 1, img_size, img_size).cuda()) 
                out_imgs = torch.cat(out_list)
                save_image(out_imgs, pj(img_fd, "%05d.png" % epoch), nrow=dataset.seq_len+2, normalize=True)
                
            batches_done += 1


def val(img_size = 64 ,channels = 1, batch_size = 8, latent_dim = 100 , embed_out = 10, cuda_id=1, data_root="/home/jiangxt18/Data/SN_C",band="B3", iter_max = -1):

    
    torch.cuda.set_device(cuda_id)
    cuda = True if torch.cuda.is_available() else False

    dataset = load_ds(img_size=img_size,band=band)

    # dataset = ImgDS_Seq(root_path="/home/jiangxt18/Data/FD", 
        # img_size=img_size,channels=1, seq_len=3)

    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
    )

    _root_path = os.path.abspath(os.path.dirname(__file__))
    data_head="%s_%d_%d_%d_%d"%(band,img_size,batch_size,latent_dim,embed_out)

    # _root_path = os.path.abspath(os.path.dirname(__file__))
    # data_head = data_root.split("/")[-1]

    img_fd = pj(_root_path,"images", data_head)
    cp_fd = pj(_root_path,"checkpoints", data_head)
    
    for _,_,files in os.walk(cp_fd):
        break
    files.sort()

    for f in files:
        cp_path = pj(cp_fd, f)
        # generator = Generator(img_size, latent_dim, channels)
        generator= torch.load(cp_path)
        generator.eval()

        if cuda:
            generator.cuda()
        out_list = [] 
        for i, (img_seq, next_img, block_id) in enumerate(dataloader):
            if cuda:
                img_seq, next_img, block_id= img_seq.cuda(), next_img.cuda(),block_id.type(torch.LongTensor).cuda()
            img_seq, next_img, block_id = Variable(img_seq), Variable(next_img), Variable(block_id)
            out = generator(img_seq,block_id)
            img_seq = img_seq.view(img_seq.size(1),img_seq.size(2),img_seq.size(3),img_seq.size(4))
            out_list.append(img_seq)
            out_list.append(next_img)
            out_list.append(out)
            if iter_max ==-1:
                continue
            elif i> iter_max:
                break
        out_imgs=torch.cat(out_list).detach().cpu()
        save_image(out_imgs, pj(img_fd, "wganLstmVal%d_%s.png" % (img_size,f.split(".")[0])), nrow=dataset.seq_len + 2, normalize=True)
        del generator


def exp_config(img_size = 128 ,channels = 1,n_epochs =20001, batch_size = 32, latent_dim = 200 ,sample_interval=2000,  embed_out = 255, cuda_id=0, band="B3"):
    train(img_size = img_size ,channels = channels,n_epochs =n_epochs, batch_size = batch_size, latent_dim = latent_dim,  embed_out = embed_out ,sample_interval=sample_interval, cuda_id=cuda_id, band=band)
    # val(img_size=img_size, channels=channels, batch_size = batch_size, latent_dim = latent_dim, embed_out = 10, cuda_id=cuda_id, band=band,iter_max=40)
    pass

if __name__ == "__main__":
    exp_config(band="B3")   
