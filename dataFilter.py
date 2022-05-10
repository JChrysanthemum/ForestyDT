import os 
import tarfile
from os import listdir
from os.path import isfile
import shutil
from os.path import join as pj
from qgisWarpper import start_app, clip_raster_by_vector, fill_gap, merge_rasters, convert_to_tiff
import numpy as np

Data_root = r"E:\Data\QL"

def _list_files(_path):
    return [f for f in listdir(_path) if isfile(pj(_path, f))]

def raw_data_check( _root_raw = r"E:\Data\RawData"):
   
    data_dict = {}

    files = _list_files(_root_raw)
    # print(files)
    for f in files:
        f_l = f.split("_")
        suffix = f.split(".")[-1]
        if suffix != "tar":
            continue

        k = f_l[3][:4] # year as key

        # # @todo delete
        # if f_l[2] !="133034":
        #     continue

        # k = f_l[2] # loc as key

        if k not in data_dict:
            data_dict[k]=[f]
        else:
            data_dict[k].append(f)
    for k,v in data_dict.items():
        print(k, len(data_dict[k]))
        # continue
        dir_root = pj(Data_root, k)
        if not os.path.exists(dir_root):
            os.mkdir(dir_root)
        for f in v:
            shutil.copy(pj(_root_raw, f), pj(dir_root, f))

def decompress():
    import tarfile
    for _, fds, _ in os.walk(Data_root):
        break
    # print(fds)

    for fd in fds:
        print("Decompressing ",fd)
        _root = pj(Data_root,fd)
        _root_cp = pj(_root,"raw")
        if not os.path.exists(_root_cp):
            os.mkdir(_root_cp)
            # shutil.rmtree(_root_cp)
        # os.mkdir(_root_cp)
        files = _list_files(_root)
        for f in files:
            my_tar = tarfile.open(pj(_root,f))
            my_tar.extractall(_root_cp) # specify which folder to extract to
            my_tar.close()
    
def clean_after_decompress():

    for _, fds, _ in os.walk(Data_root):
        break
    # print(fds)
    for fd in fds:
        _root = pj(Data_root,fd)
        files = _list_files(_root)
        for f in files:
            if f.split(".")[-1] == "tar":
                os.remove(pj(_root,f))

# raw_data_check()
# decompress()
# clean_after_decompress()

def new_release():
    changed = r"""2001 132034
    2003 132034
    2004 134033
    2005 134033
    2006 134033
    2008 133033 134033
    2009 134033
    2010 133034 134033 135033
    2011 133034
    2012 134033
    2013 134033
    2014 134033
    2015 133034
    2017 134033
    2018 133034
    2020 134033
    2021 134033"""
    new_root = r"E:\Data\new"
    old_root = r"E:\Data\RawData"
    # res={}
    for dt in changed.split('\n'):
        fl = dt.strip(" ").split(" ")
        year = fl[0]
        _root = pj(r"E:\Data\QL", year, "raw")
        files = _list_files(_root)
        tars = _list_files(old_root)
        locs = fl[1:]
        for f in files:
            for loc in locs:
                if f.find(loc)!=-1:
                    os.remove(pj(_root,f))
        for t in tars:
            for loc in locs:
                if t.find(loc)!=-1 and t.find(year)!=-1:
                    os.remove(pj(old_root,t))
    return
    raw_data_check(new_root)
    decompress()
    clean_after_decompress()

# new_release()      

def filter_to_bands():
    """
    Move all b1 - b8 tifs and fill the gapes

    """

    def save2dict(_path):
        files = _list_files(_path)
        files.sort()
        res = {}
        for f in files:
            if f.split(".")[-1] != "TIF":
                continue
            fl = f.split("_")
            loc = fl[2]
            if fl[-1][0] == "B" and fl[-1][1].isdigit():
                fname = "%s%s.TIF"%(loc,fl[-1][:2])
            elif fl[-2] == "VCID" and fl[-3] == "B6":
                fname = "%sB6.TIF"%loc
            else:
                continue
            res[fname] = f
        # a=list(res.keys()).copy()
        # a.sort()
        # print(a)
        return res


    start_app()
    for _, fds, _ in os.walk(Data_root):
        break
    # print(fds)
    # @todo delete
    fds = ["2018", "2019"]

    for fd in fds:
        _root = pj(Data_root,fd)
        _root_raw = pj(_root,"raw")
        _root_mask = pj(_root_raw,"gap_mask")

        _root_filter = pj(_root,"filter")
        if not os.path.exists(_root_filter):
            os.mkdir(_root_filter)
        #     shutil.rmtree(_root_filter)
        # os.mkdir(_root_filter)

        files = save2dict(_root_raw)

        # 1.a Check gap masl folder
        if os.path.exists(_root_mask):
            masks = save2dict(_root_mask)
            for k,v in masks.items():
                # @todo delete
                if v.find("134033") == -1 and v.find("135033") == -1 :
                    continue
                fill_gap(pj(_root_raw,files[k]), pj(_root_mask, v), pj(_root_filter, k),overwrite=True)
                # print(pj(_root_raw,files[k]), pj(_root_raw, v), pj(_root_filter, k))
                files.pop(k, None)
                
        for k,v in files.items():
            # @todo delete
            if v.find("134033") == -1 and v.find("135033") == -1 :
                continue
            shutil.copy(pj(_root_raw, v), pj(_root_filter,k))
        print("%s Done !" % fd)

def gap_fd_check():
    for _, fds, _ in os.walk(Data_root):
        break
    # print(fds)
    
    # @todo delete
    fds = ["2018", "2019"]

    for fd in fds:
        _root = pj(Data_root,fd)
        _root_raw = pj(_root,"raw")
        _root_mask = pj(_root_raw,"gap_mask")
        files = _list_files(_root_raw)
        for f in files:
            # # @todo delete
            # if f.find("133034") == -1:
            #     continue
            if f.find("GM") != -1:
                if not os.path.exists(_root_mask):
                    os.mkdir(_root_mask)
                # print(pj(_root_raw,f), pj(_root_mask,f))
                shutil.move(pj(_root_raw,f), pj(_root_mask,f))

gap_fd_check()
filter_to_bands()

def merge_locs():
    start_app()
    for _, fds, _ in os.walk(Data_root):
        break
    fds.sort()
    # @todo delete
    fds = ["2018", "2019"]


    for fd in fds:
        _root = pj(Data_root,fd)
        _root_filter = pj(_root,"filter")
        _root_merge = pj(_root,"merge")
        if os.path.exists(_root_merge):
            shutil.rmtree(_root_merge)
        os.mkdir(_root_merge)


        files = _list_files(_root_filter)
        res = {}
        for f in files:
  
            key = f[-6:-4] # B?
            f = pj(_root_filter,f)
            if key in res:
                res[key].append(f)
            else:
                res[key] = [f]
        # print(res)

        for k,v in res.items():
            temp_suffix = pj(_root_merge, "_temp_%s"%k)
            # print("merge_rasters", v, temp_suffix)
            print("Mergeing %s %s to temp sdat"%(fd, k))
            merge_rasters(v, temp_suffix)
            # print("convert_to_tiff", "%s.sdat"%temp_suffix , pj(_root_merge,"%s.TIF"%k)) 
            # print("Convert temp sdat to file %s" % pj(_root_merge,"%s.TIF"%k))
            # convert_to_tiff("%s.sdat"%temp_suffix , pj(_root_merge,"%s.TIF"%k))
        
        # continue
        # files = _list_files(_root_merge)
        # print("Delete temp sdat files, waiting")
        # for f in files:
        #     if f[:7] == "_temp_B":
        #         os.remove(pj(_root_merge, f))


def locs2tiffs():
    start_app()
    for _, fds, _ in os.walk(Data_root):
        break
    fds.sort()
    # @todo delete
    fds = ["2018", "2019"]


    for fd in fds:
        _root = pj(Data_root,fd)
        _root_merge = pj(_root,"merge")
        files = _list_files(_root_merge)
        print("Converting %s"%fd)
        for f in files:
            fl = f.split(".")
            if fl[-1].upper() == "TIF":
                continue
            # if fl[0][:7] != "_temp_B":
            #     continue
            if fl[-1] == "sdat":
                suffix = fl[0][-2:]
                convert_to_tiff(pj(_root_merge, f), pj(_root_merge, "%s.TIF"%suffix))
            os.remove(pj(_root_merge, f))


merge_locs()

locs2tiffs()

