import os 
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, r'/home/jiangxt21/Project/gansLstm/data')
from QMNP import QMNP_Seq, QMNP_Seq_Block, SeqDS,QMNP_Seq_Part

import shutil
from os import listdir
from os.path import isfile, isdir
from os.path import join as pj
import torch

def list_files(_path):
    return [f for f in listdir(_path) if isfile(pj(_path, f))]

def list_folders(_path):
    return [f for f in listdir(_path) if isdir(pj(_path, f))]

def mkdir2(_path,renew=False):
    if os.path.exists(_path):
        if renew:
            shutil.rmtree(_path)
        else:
            return
    os.mkdir(_path)

def func_parm_num(func):
    return func.__code__.co_argcount + func.__code__.co_kwonlyargcount

class SampleInput():
    def __init__(self,batch_size=2,img_size=128,seq_len=4,channels=1) -> None:
        
        self.img_seq=torch.rand(batch_size, seq_len, channels, img_size, img_size)
        self.next_img=torch.rand(batch_size, channels, img_size, img_size)
        self.nid=torch.rand(batch_size)