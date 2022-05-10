import cv2
import os
from os import listdir
from os.path import isfile
from os.path import join as pj
from qgisWarpper import start_app, clip_raster_by_vector, fill_gap, start_app_grass, cloud_mask,merge_masks
import numpy as np
import time
PATH_BOUND = r"D:\Data\Forestry\Landsat\Desert-Oasis\Boundary\Fixed"
PATH_YEARS = r"E:\Data\QL"
PATH_NEW = r"D:\Data\Forestry\Landsat2"

def _list_files(_path):
    return [f for f in listdir(_path) if isfile(pj(_path, f))]


def filter(_files_root):
    """
    Fill tiff to dict

    Return tifs, tars

    Setting :
    Location LLLPPP = 133033, enough for the roi
    Seneor: LT > LE,

    :return: dict with keys[b1-b7], and corresponding values[path1-path7]
    """

    for _,fd,files in os.walk(_files_root,topdown=False):
        pass
    tifs={}
    tars={}

    for f in files:
        res = f.split("_")
        if len(res) <=2:
            continue
        fname = res[0][:2] + res[-1].split(".")[0]
        if res[-1][0]!="B" or not res[-1][1].isdigit():
            continue
        if res[-1][-3:] == ".gz":
            tars[fname] = f
        if res[-1][-3:] == "TIF"and res[2]=="133033":
            tifs[fname] = f
    # print(tifs.values())
    # print(tars)
    return tifs, tars

def process_gap():
    start_app()

    for _, fds, _ in os.walk(PATH_YEARS):
        break
    years = []
    for fd in fds:
        if fd.isdigit():
            years.append(fd)

    for year in years:
        _root = pj(PATH_YEARS,year)
        for _, fds, _ in os.walk(_root):
            break
        if "gap_mask" in fds:
            _root_fix = _root+"-fixed"
            _root_mask = pj(_root,"gap_mask")
            _, masks = filter(_root_mask)
            tifs, _ = filter(_root)
            # print(_root)
            # print(masks.keys())
            # print(tifs.keys())
            # print(_root_fix)
            if not os.path.exists(_root_fix):
                os.mkdir(_root_fix)
            for k,v in masks.items():
                mask_file = v
                raster_file = tifs[k]
                fill_gap(pj(_root,raster_file), pj(_root_mask,mask_file), pj(_root_fix, raster_file))
                # print(mask_file, raster_file)
            # os.mkdir(_root_fix)

def cut_all(overwrite=True):
    # start_app()
    mask = r"E:\Data\Shapes\QMNP_clip.shp"
    for _, fds, _ in os.walk(PATH_YEARS):
        break
    years = []
    for fd in fds:
        if fd.isdigit():
            years.append(fd)

    for year in years:
        _root = pj(PATH_YEARS,year,"merge")
        _root_new = pj(PATH_YEARS,year,"cliped")
        
        tifs = _list_files(_root)
        if len(tifs) == 0:
            continue
        if not os.path.exists(_root_new):
            os.mkdir(_root_new)

        for f in tifs:
            clip_raster_by_vector(pj(_root,f), mask, pj(_root_new,f), overwrite=overwrite)
        # print(tifs.keys())

_mbx_angle = 122.308720
def rotate_cut(img_name):
    def crop_minAreaRect(img, rect):
        # rotate img
        angle = rect[2]
        # print(rect)
        rows,cols = img.shape[0], img.shape[1]
        M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
        img_rot = cv2.warpAffine(img,M,(cols,rows),borderValue =0)

        # rotate bounding box
        rect0 = (rect[0], rect[1], angle) 
        box = cv2.boxPoints(rect0)
        pts = np.int0(cv2.transform(np.array([box]), M))[0]    
        pts[pts < 0] = 0

        # crop
        img_crop = img_rot[pts[1][1]:pts[0][1], 
                           pts[1][0]:pts[2][0]]
        
        # cv2.imshow("rot",cv2.resize(np.rot90(img_crop), (500,500)))
        # cv2.waitKey(0)
        # exit()

        return img_crop
    gray = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE) #'D:\Data\Forestry\Landsat2\1987\B1.TIF'
    # time.sleep(5)
    # print(img_name,gray)
    pad_w = int(0.9 * gray.shape[0])
    pad_h = int(0.9 * gray.shape[1])
    # print(pad_w,pad_h)
    padding = cv2.copyMakeBorder(gray, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value = 0)

    ret, binary = cv2.threshold(padding,0,255,cv2.THRESH_BINARY)  
    # cv2.imshow("pad",cv2.resize(binary, (300,500)))
    # cv2.waitKey(0)
    # exit()
    # binary = 255 - binary
    contours,_ = cv2.findContours(binary, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key = cv2.contourArea)

    # default 0
    # @todo choose mask counter
    print(len(contours))
    rect = cv2.minAreaRect(c)

    # cv2.drawContours(binary,contours,-1,125,100)
    # print(rect)
    # cv2.imshow("pad",cv2.resize(binary, (300,500)))
    # cv2.waitKey(0)
    # exit()

    # cv2.imshow("binary",cv2.resize(binary, (300,500)))
    # cv2.waitKey(0)
    # exit()

    img_croped = crop_minAreaRect(padding, rect)
    # cv2.imshow("s",cv2.resize(np.rot90(img_croped), (500,300)))
    # cv2.waitKey(0)
    # exit()

    return np.rot90(img_croped)

def rotate_all(overwrite=True):

    start_app()
    for _, fds, _ in os.walk(PATH_YEARS):
        break
    years = []
    for fd in fds:
        if fd.isdigit():
            years.append(fd)
    years = ["2001"]
    for year in years:
        _root = pj(PATH_YEARS ,year,"cliped")
        _root_new = pj(PATH_YEARS ,year, "rotated")
        if not os.path.exists(_root_new):
            os.mkdir(_root_new)
        tifs = _list_files(_root)
        if len(tifs) == 0:
            continue
        for input_file in tifs:
            output_file = rotate_cut(pj(_root,input_file))
            cv2.imwrite(pj(_root_new,input_file), output_file)
            # print(pj(_root_new,input_file))

def tightBound_cut():
    def crop_minAreaRect(img, rect):
        # rotate img
        angle = rect[2]
        rows,cols = img.shape[0], img.shape[1]
        M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
        img_rot = cv2.warpAffine(img,M,(cols,rows))

        # rotate bounding box
        rect0 = (rect[0], rect[1], angle) 
        box = cv2.boxPoints(rect0)
        pts = np.int0(cv2.transform(np.array([box]), M))[0]    
        pts[pts < 0] = 0

        # crop
        img_crop = img_rot[pts[1][1]:pts[0][1], 
                           pts[1][0]:pts[2][0]]

        return img_crop
    
    padding_img_name = r"D:\Data\Forestry\Landsat2\1986\B1.TIF"
    target_img_name = r"D:\Projects\2021\qgis\roi.TIF"

    gray = cv2.imread(padding_img_name, cv2.IMREAD_GRAYSCALE) #'D:\Data\Forestry\Landsat2\1987\B1.TIF'
    img = cv2.imread(target_img_name, cv2.IMREAD_GRAYSCALE)

    padding = cv2.copyMakeBorder(gray, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value = 255)
    img = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value = 255)

    ret, binary = cv2.threshold(padding,254,255,cv2.THRESH_BINARY)  
    binary = 255 - binary
    contours,_ = cv2.findContours(binary, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    rect = cv2.minAreaRect(contours[0])

    # cv2.drawContours(binary,contours,-1,125,3)
    # cv2.imshow("counter",binary)
    # cv2.waitKey(0)

    img_croped = crop_minAreaRect(padding, rect)
    cv2.imshow("s",np.rot90(img_croped))
    cv2.waitKey(0)
    return np.rot90(img_croped)

def mask_cloud(overwrite=True):

    start_app_grass()
    for _, fds, _ in os.walk(PATH_YEARS):
        break
    years = []
    for fd in fds:
        if fd.isdigit():
            years.append(fd)
    # years = ["2001"]
    for year in years:
        _root = pj(PATH_YEARS ,year,"raw")
        _root_new = pj(PATH_YEARS ,year, "cloud")
        if not os.path.exists(_root_new):
            os.mkdir(_root_new)
        files = _list_files(_root)
        masks=[]
        for f in files:
            if f.find("QA_PIXEL")!=-1:
                loc = f.split('_')[2]
                cloud_mask(pj(_root,f),pj(_root_new,'%s.TIF'%loc))
                masks.append(pj(_root_new,'%s.TIF'%loc))
                # qa_bits[loc]=f
        merge_masks(masks, pj(_root_new,'merged.TIF'))
        # print(len(qa_bits))
        # exit()

def merge_clouds():
    start_app_grass()
    for _, fds, _ in os.walk(PATH_YEARS):
        break
    years = []
    for fd in fds:
        if fd.isdigit():
            years.append(fd)
    for year in years:
        _root = pj(PATH_YEARS ,year, "cloud")
        files = _list_files(_root)
        masks=[]
        for f in files:
            if f.split('.')[-1]=="TIF" and f.find("merged")==-1:
                masks.append(pj(_root,f))
                # qa_bits[loc]=f
        print(merge_masks(masks, pj(_root,'merged.TIF')))

def _rotate_specific(fname=r"D:\Data\Forestry\QMNP\GFC_2014_roi.png"):
    img = rotate_cut(fname)
    sav = r"D:\Data\Forestry\QMNP\GFC_2014_roi_cliped.png"
    # cv2.imshow("winname", img)
    # cv2.waitKey(0)
    cv2.imwrite(sav, img)


# img = cv2.imread(r"D:\Projects\2021\qgis\test2.TIF", cv2.IMREAD_GRAYSCALE)
# print(255-img[:2,:2])


if __name__ == "__main__":
    # filter(r"D:\Data\Forestry\Landsat\Desert-Oasis\1986")
    start_app()
    # cut_all()
    # rotate_all()

    # _rotate_specific()
    # Roate Angle -62.88364028930664
    # start_app()
    # clip_raster_by_vector(input_raster=r"D:\Data\Forestry\QMNP\GFC_2010_bin.tif", 
    # # input_vector=r"E:\Data\Shapes\QMNP_clip_diff.shp",
    # input_vector=r"E:\Data\Shapes\QMNP_clip.shp",
    # output_raster=r"D:\Data\Forestry\QMNP\GFC_2010_roi.TIF",overwrite=True)
    # img = rotate_cut()

    # mask_cloud()

    # merge_clouds()