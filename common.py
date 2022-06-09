from os.path import join as pj
from os import listdir
from os.path import isfile, isdir
import shutil
import os
import pickle

def _list_files(_path):
    return [f for f in listdir(_path) if isfile(pj(_path, f))]

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
    os.makedirs(_path)

def save_pickle(fname, obj):
    f= open(fname,"wb")
    pickle.dump(obj, f)
    f.close()

def load_pickle(fname):
    f= open(fname,"rb")
    res = pickle.load(f)
    f.close()
    return res
