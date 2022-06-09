import os
from os.path import join as pj
import torch
import shutil
from torch import cuda
from torch.autograd import Variable
from torchvision.utils import save_image
import numpy as np
import os
from pathlib import Path

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="2"

if __name__ == "__main__":
    from dataWrapper import *
    from models import *
else:
    from .dataWrapper import *
    from models import *
    
_root_path = Path(os.path.dirname(os.path.realpath(__file__))).parent.absolute()

def train(img_size = 128 ,n_epochs =101, batch_size = 8, latent_dim = 100 ,
            sample_interval=5, cuda_id=0, data_config=None, model_config=None):
    
   
    dataset,data_suffix = data_config
    ex_suffix = dataset.name
    dataset = dataset.train_set
    model_name, model = model_config[:2]
    optimizer = model_config[2](model.parameters(), *model_config[3])
    

    band = dataset.band
    loss = torch.nn.MSELoss()

    # Configure data loader)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )


    global _root_path
    root_path = pj(_root_path,"result",data_suffix, model_name)
    if not os.path.exists(root_path):
        os.makedirs(root_path, exist_ok=True)
    data_head="[%s]:%d_%s_%d_%d"%(ex_suffix,img_size,band,batch_size,latent_dim)

    img_fd = pj(root_path,"images", data_head)
    cp_fd = pj(root_path,"checkpoints", data_head)
    if os.path.exists(img_fd):
        shutil.rmtree(img_fd)
        print(img_fd)
    if os.path.exists(cp_fd):
        shutil.rmtree(cp_fd)
        print(cp_fd)


    os.makedirs(img_fd)
    os.makedirs(cp_fd)

    torch.cuda.set_device(cuda_id)
    cuda = True if torch.cuda.is_available() else False
    print("Use gpu: ",cuda, cuda_id)

    if cuda:
        model.cuda()
        loss.cuda()

    sample_interval *=len(dataloader)

    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    # ----------
    #  Training
    # ----------
    batches_done = 0
    for epoch in range(n_epochs):
        for i, (img_seq, next_img, nid) in enumerate(dataloader):
            # batch_size = imgs.shape[0]
            if len(img_seq) <= 1:
                continue
            
            if cuda:
                img_seq, next_img, nid = img_seq.cuda(), next_img.cuda(), nid.cuda()
            img_seq, next_img, nid= Variable(img_seq.type(FloatTensor)), Variable(next_img), Variable(nid.reshape(nid.size(0)))

            # Configure input
            real_imgs = Variable(next_img.type(FloatTensor))
            labels = Variable(nid.type(LongTensor))

            optimizer.zero_grad()

            # Generate a batch of images
            gen_imgs = model(img_seq, labels)

            g_loss = loss(gen_imgs,real_imgs)

            g_loss.backward()
            optimizer.step()

            # print("[Epoch %d/%d] [Batch %d/%d] [G loss: %f]" % (epoch, n_epochs, i, len(dataloader), g_loss.item()))

            if batches_done % sample_interval == 0 and batches_done != 0:
                torch.save(model, "%s/%d.pt" % (cp_fd, epoch))
                iter_max = batch_size if batch_size <= len(gen_imgs.data) else len(gen_imgs.data)
                out_list = []
                for i in range(iter_max):
                    out_list.append(img_seq[i].view(dataset.seq_len, 1, img_size, img_size))
                    out_list.append(next_img[i].view(1, 1, img_size, img_size).cuda())
                    out_list.append(gen_imgs.data[i].view(1, 1, img_size, img_size).cuda()) 
                out_imgs = torch.cat(out_list)
                save_image(out_imgs, pj(img_fd, "%05d.png" % epoch), nrow=dataset.seq_len+2, normalize=True)
            batches_done += 1

def out(img_size = 128 , batch_size = 8, latent_dim = 100 , cuda_id=0, data_config=None, model_config=None, clean=False):
    global _root_path
    dataset,data_suffix = data_config
    ex_suffix = dataset.name
    dataset = dataset.test_set
    model_name, _ = model_config[:2]

    band = dataset.band
    # Configure data loader)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
    )

    root_path = pj(_root_path,"result",data_suffix, model_name)
    if not os.path.exists(root_path):
        os.makedirs(root_path,exist_ok=True)
    data_head="[%s]:%d_%s_%d_%d"%(ex_suffix,img_size,band,batch_size,latent_dim)
    save_root = pj(root_path,"output",band)

    if not os.path.exists(save_root):
        os.makedirs(save_root)
    out_fd = pj(save_root, data_head)
    if not os.path.exists(out_fd):
        os.makedirs(out_fd)
    cp_fd = pj(root_path, "checkpoints",data_head)
    files = list_files(cp_fd)
    cps = []
    for f in files:
        if int(f.split(".")[0])==0:
            continue
        cps.append(f)

    torch.cuda.set_device(cuda_id)
    cuda = True if torch.cuda.is_available() else False
    print("Use gpu: ",cuda, cuda_id)
    iter_max=-1

    for cp in cps:
        generator=torch.load(pj(cp_fd,cp))
        generator.eval()
        p_root = pj(out_fd, cp.split(".")[0])
        if not os.path.exists(p_root):
            os.makedirs(p_root)
        if cuda:
            generator.cuda()
        for i, (img_seq, next_img, block_id) in enumerate(dataloader):
            if cuda:
                img_seq, next_img, block_id= img_seq.cuda(), next_img.cuda(),block_id.type(torch.LongTensor).cuda()
            img_seq, next_img, block_id = Variable(img_seq), Variable(next_img), Variable(block_id)
            out = generator(img_seq,block_id)
            save_image(out, pj(p_root, "%4d.png" % i), nrow=1, normalize=True)
            if iter_max ==-1:
                continue
            elif i> iter_max:
                break     
        del generator

    if clean:
        shutil.rmtree(pj(root_path,"checkpoints", data_head))
        shutil.rmtree(pj(root_path,"images", data_head))

def main(img_size = 128,n_epochs=501,batch_size=16,latent_dim=100,sample_interval=500,cuda_id=0,clean=False):
    # bands = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8"]
    bands = ["B2","B3","B4"]
    # sample_interval = n_epochs-1
    
    # Blocks in one view of study area
    features = QMNP_Seq(img_size=img_size,band=bands[0]).features
    data_suffix = "P"
    # model_config = ["CLC", Models.Conv_LSTM_Conv(img_size,latent_dim), *(Optims.RMSprop.value)]
    model_config = ["WGAN", Models.Conv_LSTM_WGAN(img_size,latent_dim), *(Optims.RMSprop.value)]
    # model_config = ["VAE_CGAN", Models.VAE_CGAN(img_size,latent_dim,features), *(Optims.Adam.value)]
    # model_config = ["VAE_WGAN", Models.VAE_WGAN(img_size,latent_dim), *(Optims.Adam.value)]
    print(model_config[0])
    root_path = pj(_root_path,"result",data_suffix, model_config[0])

    # if clean:
    #     _save_dirs = [pj(root_path,s) for s in ["checkpoints","images"]]
    #     for _sd in _save_dirs:
    #         if os.path.exists(_sd):
    #             shutil.rmtree(_sd)
    #         os.makedirs(_sd)
    
    for band in bands:
        
        # Divide length.
        d_len = 15
        
        itx = features//d_len 
        for i in range(itx):
            dataset = QMNP_Seq_Part(img_size=img_size,band=band,blks=list(range(i*d_len,(i+1)*d_len)),name="%d-%d"%(i*d_len,(i+1)*d_len))
            
            data_config = [dataset, data_suffix]
            
            train(img_size=img_size,n_epochs=n_epochs,batch_size=batch_size,latent_dim=latent_dim,sample_interval=sample_interval,cuda_id=cuda_id,data_config=data_config, model_config=model_config)
            out(img_size = img_size , batch_size = batch_size, latent_dim = latent_dim , cuda_id=cuda_id, data_config=data_config, model_config=model_config,clean=True)
            # exit()


if __name__=="__main__":
    main(clean=True)
