if __name__ == "__main__":
    from .convLstm import ConvLstm
    from .dataWrapper import *
    from .generators import *
    from .vae import VariationalEncoder
else:
    from convLstm import ConvLstm
    from dataWrapper import *
    from generators import *
    from vae import VariationalEncoder


from enum import Enum

# print(func_parm_num(WGAN_Generator.forward))
# print(func_parm_num(CGAN_Generator.forward))

class Combined(nn.Module):
    def __init__(self, img_size=128, latent_dim=100,
                    n_classes=-1, model_hid=CGAN_Generator, model_hat=ConvLstm) -> None:
        super().__init__()
        channels=1
        self.cnnlstm = model_hat(img_size=img_size, latent_dim=latent_dim)
        if n_classes ==-1:
            self.generator = model_hid(img_size=img_size, channels=channels,latent_dim=latent_dim)
            if func_parm_num(model_hid.forward) == 3:
                raise ValueError("N classes not set for embedding")
            else:
                self.label_emb = None
                self.forward = self.forward_no_condition
        else :
            self.generator = model_hid(img_size=img_size, channels=channels,latent_dim=latent_dim,n_classes=n_classes)
            if func_parm_num(model_hid.forward) ==2:
                print("Unneed param nclasses")
            else:
                self.label_emb = nn.Embedding(n_classes, latent_dim)
                self.forward = self.forward_with_condition        
    
    def forward_no_condition(self, x, dummy):
        z = self.cnnlstm(x)
        img = self.generator(z)
        return img
    
    def forward_with_condition(self, x, labels):
        z = self.cnnlstm(x)
        # gen_input = torch.cat((self.label_emb(labels), z), -1)
        img = self.generator(z, labels)
        return img

# class Models(Enum):
class Gens(Enum):
    WGAN = WGAN_Generator
    CGAN = CGAN_Generator
    COGAN = CONWGAN_Generator

class Models(Enum):
    Conv_LSTM_CGAN = lambda img_size, latent_dim, n_classes: Combined(img_size=img_size,latent_dim=latent_dim,model_hid=CGAN_Generator,model_hat=ConvLstm, n_classes=n_classes)
    Conv_LSTM_Conv = lambda img_size, latent_dim: Combined(img_size=img_size,latent_dim=latent_dim,model_hid=DeConv,model_hat=ConvLstm)
    Conv_LSTM_WGAN = lambda img_size, latent_dim: Combined(img_size=img_size,latent_dim=latent_dim,model_hid=WGAN_Generator,model_hat=ConvLstm)
    VAE_CGAN = lambda img_size, latent_dim, n_classes: Combined(img_size=img_size,latent_dim=latent_dim,model_hid=CGAN_Generator,model_hat=VariationalEncoder, n_classes=n_classes)
    VAE_WGAN = lambda img_size, latent_dim: Combined(img_size=img_size,latent_dim=latent_dim,model_hid=WGAN_Generator,model_hat=VariationalEncoder)


class Optims(Enum):
    RMSprop = [torch.optim.RMSprop, [0.001]] # lr
    Adam = [ torch.optim.Adam,[0.0002,(0.5,0.999)]] # lr (b1,b2)
