from os.path import join as pj
from os import listdir
from os.path import isfile, isdir
import shutil
import os

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
    os.mkdir(_path)
