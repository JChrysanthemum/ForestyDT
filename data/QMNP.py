import os
from os.path import join as pj
from os import SEEK_CUR, listdir
from os.path import isfile, isdir
from PIL import Image
import shutil
from numpy.lib.npyio import load
import torch
import torchvision
import torchvision.transforms as transforms
import pickle
import numpy as np

def list_files(_path):
    return [f for f in listdir(_path) if isfile(pj(_path, f))]

def list_folders(_path):
    return [f for f in listdir(_path) if isdir(pj(_path, f))]

DATA_ROOT = r"/home/jiangxt21/synData/QMNP_Block"
# DATA_ROOT = r"D:\Data\Forestry\QMNP_Block"
QMNP_ROOT = {
    64: pj(DATA_ROOT,"64"),
    128: pj(DATA_ROOT,"128"),
    256: pj(DATA_ROOT,"256")
}

def load_raw(img_size = 64, test_years=[2021], seq_len=4):
    _root = QMNP_ROOT[img_size]
    years=list_folders(_root)
    years.sort()

    # todo add val
    def get_train_val_year():
        idx = [years.index(str(y)) for y in test_years]
        _start = 0
        res_train = []
        for ed in idx:
            if len(years[_start:ed])<seq_len+1:
                print("From %d to %d length not available" %(years[_start],years[ed]))
                continue
            res_train.append(years[_start:ed])
            _start=ed+1
        if idx[-1] != len(years)-1:
            if len(years[idx[-1]:-1])<seq_len+1:
                print("From %s to %s length not available" %(years[_start],years[ed]))
            else:
                res_train.append(years[idx[-1]:-1])

        res_test = [years[y-4:y+1] for y in idx]
        

        
        return res_train,res_test
    
    train_list, test_list = get_train_val_year()
    # print(train_list, test_list)
    # exit()
    
    # for _,bands,_ in os.walk(pj(_root, years[0])):
    #     break
    bands = list_folders(pj(_root, years[0]))
    # bands = ["B3","B4","B5"]

    files = list_files(pj(_root, years[0], bands[0]))
    
    # print(years)
    bands.sort()
    files.sort()

    ds_len = len(files)
    res = {
        "train":{},
        "test":{}
    }
    for bd in bands:
        res["train"][bd]=[]
        res["test"][bd]=[]
        print("Reading Band %s"%bd)
        for f in files:
            for y_list in train_list:
                f_list = []
                for year in y_list:
                    img = Image.open(pj(_root, year, bd, f)).convert('L')
                    if img.size != (img_size,img_size):
                        raise Exception("Error Found, image size in consist with " +  (img_size,img_size))
                    f_list.append(img)
                res["train"][bd].append(f_list)
            for y_list in test_list:
                f_list = []
                for year in y_list:
                    img = Image.open(pj(_root, year, bd, f)).convert('L')
                    if img.size != (img_size,img_size):
                        raise Exception("Error Found, image size in consist with " +  (img_size,img_size))
                    f_list.append(img)
                res["test"][bd].append(f_list)
            
    
    f=open(pj(DATA_ROOT,'%d.rawdata'%img_size), "wb")
    pickle.dump([res,len(files)],f)
    f.close()

    # print(len(files))
    return res, len(files)

def _load_pk(_name):
    if not os.path.exists(_name):
        return None
    f = open(_name,"rb")
    res = pickle.load(f)
    f.close()
    print("DS %s load successful %s"%(type(res),_name))
    return res

def load_ds(img_size = 64, band="B1"):
    save_name = pj(DATA_ROOT,r'%d_%s.ds'%(img_size,band))
    return _load_pk(save_name)

def load_ds_blk(img_size = 64, band="B1", blk=1):
    save_name = pj(DATA_ROOT,'%d-%d_%s.ds'%(blk,img_size,band))
    return _load_pk(save_name)

def load_ds_part(img_size = 64, band="B1", name="p1"):
    save_name =  pj(DATA_ROOT,'%s-%d_%s.ds'%(name,img_size,band))
    return _load_pk(save_name)

class SeqDS(torch.utils.data.Dataset):
        def __init__(self,img_size,seq_len,band):
            self.img_size = img_size
            self.X = []
            self.X_idx = []
            self.Y = []
            self.seq_len = seq_len
            self.band= band
            self.features = 0
            super().__init__()


        def __getitem__(self, index):
            _seq = self.X[self.X_idx[index]:self.X_idx[index]+self.seq_len]
            _next =  self.X[self.X_idx[index]+self.seq_len]
            _seq_tensor = torch.Tensor(self.seq_len, 1, self.img_size, self.img_size)
            torch.stack(_seq, out=_seq_tensor)
            return _seq_tensor, _next, self.Y[index]

        def __len__(self):
            return len(self.X_idx)

class QMNP_Seq():
    def __init__(self, img_size = 128, band="B3", seq_len = 4, save=False) -> None:
        
        super().__init__()
        band_list = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "ALL"]
        if img_size not in [64,128,256]:
            raise Exception("Image size invalid")
        if band not in band_list:
            raise Exception("Band invalid")

        print("Data set initialized with [Size %d, Band %s]"%(img_size, band))

        self.train_set = SeqDS(img_size, seq_len,band)
        self.test_set = SeqDS(img_size, seq_len,band)
        self.features=0
        # print(self.train_set.band)

        f = load_ds(img_size,band)
        if not save and f != None:
            for k in f.__dict__.keys():
                setattr(self, k, getattr(f, k))
            # print(self.train_set.band)
            self.train_set.features = self.features
            self.test_set.features = self.features
            print("Load form exist file")
            print("Train set Load images: %d/%d, timesteps %d get %d sequences"%
            (len(self.train_set.X),len(self.train_set.Y), seq_len, len(self.train_set.X_idx)))
            print("Test set Load images: %d/%d, timesteps %d get %d sequences"%
            (len(self.test_set.X),len(self.test_set.Y), seq_len, len(self.test_set.X_idx)))
            return
        

        f = open(pj(DATA_ROOT,'%d.rawdata'%img_size),"rb")
        raw_data, self.features = pickle.load(f)
        f.close()
        self.train_set.features = self.features
        self.test_set.features = self.features
        # print(self.features)
        X_transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size), transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.5), (0.5)),
                # transforms.Normalize(mean, std),
            ]
        )


        for suf,X,X_idx,Y in zip(["train","test"],[self.train_set.X,self.test_set.X],[self.train_set.X_idx,self.test_set.X_idx],[self.train_set.Y,self.test_set.Y]):
            idx = 0
            if band == "ALL":
                for k,v in raw_data[suf].items():
                    if k not in [ "B3", "B4", "B5"]:
                        continue
                    band_index = band_list.index(k)
                    for i in range(len(v)):

                        year_data = v[i]
                        X += year_data
                        # Y += [[i,band_index]]*len(year_data)
                        Y += [[i]]*len(year_data)
                        X_idx += range(idx, idx + len(year_data) - seq_len)
                        idx += len(year_data)
            else:
                band_index = band_list.index(band)
                for i in range(len(raw_data[suf][band])):
                    year_data = raw_data[suf][band][i]
                    X += year_data
                    # print(len(year_data))
                    # Y += [[i,band_index]]*len(year_data)
                    Y += [[i]]*len(year_data)
                    X_idx += range(idx, idx + len(year_data) - seq_len)
                    idx += len(year_data)
            for i in range(len(X)):
                X[i] = X_transform(X[i])
            for i in range(len(Y)):
                Y[i] = torch.LongTensor(Y[i])

        
        print("Train set Load images: %d/%d, timesteps %d get %d sequences"%
        (len(self.train_set.X),len(self.train_set.Y), seq_len, len(self.train_set.X_idx)))
        print("Test set Load images: %d/%d, timesteps %d get %d sequences"%
        (len(self.test_set.X),len(self.test_set.Y), seq_len, len(self.test_set.X_idx)))
        if save:
            save_name = pj(DATA_ROOT,'%d_%s.ds'%(img_size,band))
            f = open(save_name,"wb")
            pickle.dump(self, f)
            f.close()
            print("Data set save to %s"%save_name)

class QMNP_Seq_Block():
    def __init__(self, img_size = 128, band="B3", blk=1, seq_len = 4, save=False):
            
        seq_model = QMNP_Seq(img_size,band)
        self.blk=blk
        self.train_set = SeqDS(img_size, seq_len,band)
        self.test_set = SeqDS(img_size, seq_len,band)
        self.train_set.features = seq_model.features
        self.test_set.features = seq_model.features

        f = load_ds_blk(img_size,band,blk)
        if not save and f != None:
            for k in f.__dict__.keys():
                setattr(self, k, getattr(f, k))
            
            print("Load form exist file")
            print("Train set Load images: %d/%d, timesteps %d get %d sequences"%
            (len(self.train_set.X),len(self.train_set.Y), seq_len, len(self.train_set.X_idx)))
            print("Test set Load images: %d/%d, timesteps %d get %d sequences"%
            (len(self.test_set.X),len(self.test_set.Y), seq_len, len(self.test_set.X_idx)))
            return
        
        for ori,X,X_idx,Y in zip([seq_model.train_set,seq_model.test_set],[self.train_set.X,self.test_set.X],[self.train_set.X_idx,self.test_set.X_idx],[self.train_set.Y,self.test_set.Y]):
            iter_max = len(ori.X) // seq_model.features
            # print(iter_max)
            if band == "ALL":
                raise NotImplementedError("Not work for All")
            else:
                X+=ori.X[iter_max*blk:iter_max*(blk+1)]
                Y+=[blk]*iter_max
                # for i in range(iter_max):
                #     print(i*iter_step + (i+1)*blk)
                #     X += [ori.X[i*iter_step + blk]]

                #     # print(len(year_data))
                #     # Y += [[i,band_index]]*len(year_data)
                #     Y += [blk]

                X_idx += range(0, len(X)-seq_len)
 
        print("Train set Load images: %d/%d, timesteps %d get %d sequences"%
        (len(self.train_set.X),len(self.train_set.Y), seq_len, len(self.train_set.X_idx)))
        print("Test set Load images: %d/%d, timesteps %d get %d sequences"%
        (len(self.test_set.X),len(self.test_set.Y), seq_len, len(self.test_set.X_idx)))

        if save:
            save_name = pj(DATA_ROOT,'%d-%d_%s.ds'%(blk,img_size,band))
            f = open(save_name,"wb")
            pickle.dump(self, f)
            f.close()
            print("Data set save to %s"%save_name)

class QMNP_Seq_Part():
    def __init__(self, img_size = 128, band="B3", blks=[1], seq_len = 4, save=False, name="p1"):
            
        seq_model = QMNP_Seq(img_size,band)
        self.name=name
        self.train_set = SeqDS(img_size, seq_len,band)
        self.test_set = SeqDS(img_size, seq_len,band)
        self.train_set.features = seq_model.features
        self.test_set.features = seq_model.features

        f = load_ds_part(img_size,band,name)
        if not save and f != None:
            for k in f.__dict__.keys():
                setattr(self, k, getattr(f, k))
            
            print("Load form exist file")
            print("Train set Load images: %d/%d, timesteps %d get %d sequences"%
            (len(self.train_set.X),len(self.train_set.Y), seq_len, len(self.train_set.X_idx)))
            print("Test set Load images: %d/%d, timesteps %d get %d sequences"%
            (len(self.test_set.X),len(self.test_set.Y), seq_len, len(self.test_set.X_idx)))
            return
        
        
        for ori,X,X_idx,Y in zip([seq_model.train_set,seq_model.test_set],[self.train_set.X,self.test_set.X],[self.train_set.X_idx,self.test_set.X_idx],[self.train_set.Y,self.test_set.Y]):
            idx = 0
            iter_max = len(ori.X) // seq_model.features
            for blk in blks:                
                # print(iter_max)
                if band == "ALL":
                    raise NotImplementedError("Not work for All")
                else:
                    year_data = ori.X[iter_max*blk:iter_max*(blk+1)]
                    X+= year_data
                    Y+=[blk]*iter_max
                    # X_idx += range(0, len(X)-seq_len)
                    X_idx += range(idx, idx + len(year_data) - seq_len)
                    idx += len(year_data)
                    

        print("Train set Load images: %d/%d, timesteps %d get %d sequences"%
        (len(self.train_set.X),len(self.train_set.Y), seq_len, len(self.train_set.X_idx)))
        print("Test set Load images: %d/%d, timesteps %d get %d sequences"%
        (len(self.test_set.X),len(self.test_set.Y), seq_len, len(self.test_set.X_idx)))

        if save:
            save_name = pj(DATA_ROOT,'%s-%d_%s.ds'%(name,img_size,band))
            f = open(save_name,"wb")
            pickle.dump(self, f)
            f.close()
            print("Data set save to %s"%save_name)

if __name__ == "__main__":
    img_size=128
    load_raw(128)
    # # load_raw(256)
    # for band in ["B3","B4","B5"]:
    for band in ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8"]:
    # for band in ["B3"]:    
    # # # # # for band in ["ALL"]:
        set = QMNP_Seq(img_size=img_size, band=band,save=True)
    # blk_ds = QMNP_Seq_Block(save=True,blk=0)
        # part_ds = QMNP_Seq_Part(img_size=img_size, band=band,save=True,blks=range(10),name="p1")

# todo mini cgan

# print(set[0])

# todo train val
