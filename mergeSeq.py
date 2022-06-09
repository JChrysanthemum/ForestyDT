import os,shutil
import cv2
from os import listdir
from os.path import isfile, isdir
from os.path import join as pj

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



_root = r"/home/jiangxt18/gansLstm/result/P/WGAN"
mg_fd = pj(_root,"merged")
out_fd = pj(_root,"output")
fds = list_folders(out_fd)
fds.sort()
trgs = []
for fd in fds:
    if fd.find('[')== -1 or fd.find(']') ==-1:
        continue
    rg = fd[fd.find('[')+1: fd.find(']')]
    start = int(rg.split("-")[0])
    f_list = fd.split("_")
    band = f_list[1]

    epochs = list_folders(pj(out_fd, fd))
    for eps in epochs[:2]:
        img_root = pj(out_fd, fd, eps)
        save_root = pj(mg_fd,eps,band)
        os.makedirs(save_root,exist_ok=True)
        files = list_files(img_root)
        # print(img_root)
        for f in files:
            name =int(f.split(".")[0])+start
            suffix =f.split(".")[1]
            # print("%04d.%s"%(name,suffix))

            # Without normalization           
            img = cv2.imread(pj(img_root,f),cv2.IMREAD_GRAYSCALE)
            print(img,img*255)
            exit()
            cv2.imwrite(pj(save_root,"%04d.%s"%(name,suffix)),img)

            # With normalization
            # shutil.copy(pj(img_root,f),pj(save_root,"%04d.%s"%(name,suffix)))

            # print("*%s-%s*"%(rg,band))
        
