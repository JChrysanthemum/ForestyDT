import os
import shutil
from os.path import join as pj
from os import listdir
from os.path import isfile
import cv2
import numpy as np
import pickle

from common import _list_files


DATA_ROOT=r"D:\Data\Forestry\QMNP"

def data_transfer():
    _old_root= r"E:\Data\QL"
    for _, fds,_ in os.walk(_old_root):
        break
    for fd in fds:
        if os.path.exists(pj(DATA_ROOT,fd)):
            shutil.rmtree(pj(DATA_ROOT,fd))
        shutil.copytree( pj(_old_root, fd, "rotated"), pj(DATA_ROOT,fd))

def dataset_resize():
    shp = cv2.imread(pj(DATA_ROOT,"MASK_ROI.png"),cv2.IMREAD_GRAYSCALE).shape
    for _,fds,_ in os.walk(DATA_ROOT):
        break
    res = {}
    for fd in fds:
        _root = pj(DATA_ROOT,fd)
        files = _list_files(_root)
        for f in files:
            img = cv2.imread(pj(_root,f),cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(pj(_root,f),  cv2.resize(img, (shp[1], shp[0])))

# data_transfer()
# dataset_resize()

def blocks_calc(size=256, step=32):
    mask_data = cv2.imread(r"D:\Data\Forestry\QMNP\MASK_Data.png", cv2.IMREAD_GRAYSCALE)
    mask_roi = cv2.imread(r"D:\Data\Forestry\QMNP\MASK_ROI.png", cv2.IMREAD_GRAYSCALE)
    binary_data = np.where(mask_data!=0, 1, 0).astype(np.uint8)
    binary_roi = np.where(mask_roi!=0, 1, 0).astype(np.uint8)

    h,w = mask_data.shape

    h_steps = (h-size)//(size - step)+1
    w_steps = (w-size)//(size - step)+1

    h_indice = [[i*(size-step), i*(size-step) + size] for i in range(h_steps)] + [[h-size, h]]
    w_indice = [[i*(size-step), i*(size-step) + size] for i in range(w_steps)] + [[w-size, w]]
    # print(h,w,h_indice,w_indice)

    full_data = size*size
    no_data = int(0.01*full_data)

    res_full=[]
    res_nofull=[]
    # print(mask_data.shape, mask_roi.shape)
    # exit()

    # print(h_indice, "\n", w_indice)
    for h_i in h_indice:
        for w_i in w_indice:
            area_data = binary_data[h_i[0]:h_i[1],w_i[0]:w_i[1]]
            area_roi = binary_roi[h_i[0]:h_i[1],w_i[0]:w_i[1]]
            if area_data.sum() <= no_data or area_roi.sum() <= no_data:
                pass
            elif area_data.sum() == full_data:
                res_full.append([h_i,w_i])
                binary_roi[h_i[0]:h_i[1],w_i[0]:w_i[1]]=0
            else:
                # print(area.sum())
                # print(binary_roi[h_i[0]:h_i[1],w_i[0]:w_i[1]].sum())
                if binary_roi[h_i[0]:h_i[1],w_i[0]:w_i[1]].sum() >= no_data:   
                    # print(binary_roi[h_i[0]:h_i[1],w_i[0]:w_i[1]].sum(), w_i, h_i)
                    res_nofull.append([h_i,w_i]) 
                    pass
                else:
                    pass
                    
                    
            # cv2.imshow("2", cv2.resize(binary_roi*255,(2700,550)))
            # cv2.waitKey(1)

    print("Ready to go area", len(res_full))
    print("No full area", len(res_nofull))

    res_calibar = []


    img_roi = cv2.cvtColor(binary_roi*255,cv2.COLOR_GRAY2RGB)
    img_data = cv2.cvtColor(binary_data*255,cv2.COLOR_GRAY2RGB)

    
    # cv2.imshow("ori", img_roi)
    for i in range(20):
        
        contours,_ = cv2.findContours(binary_roi, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        print("Iter %d with contours %d" % (i, len(contours)))
        if len(contours)==0:
            # print("No contours now")
            break
        
        for ct in contours:
            area = int(cv2.contourArea(ct))        
            # print(area)
            rect = cv2.minAreaRect(ct)
            box = np.int0(cv2.boxPoints(rect))
            # print(rect, type(rect))
            # print(box, type(box))
            # exit()
            start = [box[:,0].min(), box[:,1].min()]
            end = [box[:,0].max(), box[:,1].max()]
            ct_found = False
            
            if area < no_data:
                # print("a",start,end)
                cv2.fillPoly(binary_roi, pts =[ct], color=0)
                # print(binary_roi[start[1]:end[1], start[0]:end[0]])
                contours,_ = cv2.findContours(binary_roi, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                continue      

            res = None
            max_roi = 0
            # print(i)
            # print(start[1]-size+1 if start[1]-size+1 > 0 else 0, end[1]-1)
            # print(start[0]-size+1 if start[0]-size+1>0 else 0, end[0]-1)
            for h_i in range(start[1]-size+1 if start[1]-size+1 > 0 else 0, end[1]-1):
                if ct_found :
                    break
                for w_i in  range(start[0]-size+1 if start[0]-size+1>0 else 0, end[0]-1):
                    area_data = binary_data[h_i:h_i+size, w_i: w_i+size]
                    area_roi = binary_roi[h_i:h_i+size, w_i: w_i+size]
                    box2 = np.int0([[w_i+size, h_i + size], [w_i,h_i + size], [w_i,h_i],[w_i+size,h_i]])


                    # if len(contours)<=4:

                    #     cv2.imshow("1", area_data*255)
                    #     cv2.imshow("2", area_roi*255)
                    #     img_data_cp = cv2.cvtColor(binary_data*255,cv2.COLOR_GRAY2RGB)
                    #     img_roi_cp = cv2.cvtColor(binary_roi*255,cv2.COLOR_GRAY2RGB)
                    #     cv2.drawContours(img_data_cp,[box2], -1, (0,0,255), 10)
                    #     cv2.drawContours(img_roi_cp,[box2], -1, (0,0,255), 10)
                    #     cv2.imshow("data", cv2.resize(img_data_cp, (2000,400)))
                    #     cv2.imshow("roi", cv2.resize(img_roi_cp, (2000,400)))
                    #     cv2.waitKey(1)
                        

                    if area_data.sum()==full_data and area_roi.sum()>max_roi:
                       
                        # print(area_roi.sum())
                        res = [[h_i,h_i+size], [w_i,w_i+size]]
                        max_roi = area_roi.sum()
                        if area_roi.sum()==area:
                            # print("dddd")
                            ct_found = True
                            break
            if res: 
                # print("Finally")
                # print(binary_roi[res[0]:res[0]+size, res[1]: res[1]+size].sum())

                # cv2.fillPoly(binary_roi, pts =[ct], color=0)
                binary_roi[res[0][0]:res[0][1],res[1][0]:res[1][1]]=0
                res_calibar.append(res)

                contours,_ = cv2.findContours(binary_roi, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                # print(i, len(contours))
                # print(binary_roi[res[0]:res[0]+size, res[1]: res[1]+size].sum())
                # print("*"*20)
                    #    box2 = np.int0([[start[0], start[1]+size],start, [start[0]+size,start[1]],[start[0]+size,start[1]+size]])
                    #    cv2.drawContours(img_roi, [box2], -1, (255,0,0))

    print("Patched area", len(res_calibar))
    

    Blocks={
        "description":"Cubic blocks with size and step, [[height_start,height_end],[width_start,width_end]]",
        "size":size,
        "step":step,
        "normal_blocks":res_full,
        "patch_blocks":res_calibar,
    }

    f=open(r"D:\Projects\2021\qgis\Blocks_%d_%d"%(size, step),"wb")
    pickle.dump(Blocks, f)
    f.close()

# blocks_calc(size=64, step=8)
# blocks_calc(size=128, step=0)
# blocks_calc(size=256, step=32)

def block_show(size=64, step=8):
    img = cv2.imread(r"D:\Data\Forestry\QMNP\2010\combined\742.png")
    img2 = np.zeros(img.shape)
    f=open(r"D:\Projects\2021\qgis\Blocks_%d_%d"%(size, step),"rb")
    Blocks = pickle.load(f)
    f.close()

    h,w = img.shape[:2]
    _root = r"D:\Projects\2021\qgis\algor_show%d_%d.png"%(size,step)
    if not os.path.exists(_root):
        os.mkdir(_root)
    h_steps = (h-size)//(size - step)+1
    w_steps = (w-size)//(size - step)+1

    h_indice = [[i*(size-step), i*(size-step) + size] for i in range(h_steps)] + [[h-size, h]]
    w_indice = [[i*(size-step), i*(size-step) + size] for i in range(w_steps)] + [[w-size, w]]

    img_step1 = img2.copy()
    for h_i in h_indice:
        for w_i in w_indice:
            x0,x1,y0,y1 = h_i[0],h_i[1],w_i[0],w_i[1]
            box = np.int0([[y0,x0], [y0,x1], [y1,x1], [y1,x0]])
            cv2.drawContours(img_step1, [box], -1, (0,0,255),5)
    cv2.imwrite(pj(_root,"1.png"), img_step1)

    img_step2 = img2.copy()
    for res in Blocks["normal_blocks"]:
        x0,x1,y0,y1 = res[0][0],res[0][1],res[1][0],res[1][1]
        box = np.int0([[y0,x0], [y0,x1], [y1,x1], [y1,x0]])
        # print(res,box)
        cv2.drawContours(img_step2, [box], -1, (0,0,255),5)
    cv2.imwrite(pj(_root,"2.png"), img_step2)

    img_step3 = img2.copy()
    for res in Blocks["patch_blocks"]:
        x0,x1,y0,y1 = res[0][0],res[0][1],res[1][0],res[1][1]
        box = np.int0([[y0,x0], [y0,x1], [y1,x1], [y1,x0]])
        # print(res,box)
        cv2.drawContours(img_step3, [box], -1, (255,0,0),5)
    cv2.imwrite(pj(_root,"3.png"), img_step3)
    
    # img_step4 = img2.copy()
    # blcs = Blocks["normal_blocks"] + Blocks["patch_blocks"]
    # for res in blcs:
    #     x0,x1,y0,y1 = res[0][0],res[0][1],res[1][0],res[1][1]
    #     box = np.int0([[y0,x0], [y0,x1], [y1,x1], [y1,x0]])
    #     # print(res,box)
    #     cv2.drawContours(img, [box], -1, (0,0,255),5)
    #     cv2.drawContours(img_step4, [box], -1, (0,0,255),5)
    #     # cv2.imshow("2",img)
    #     # cv2.waitKey(0)
    #     # print(res,box)
    #     # exit()
    # cv2.imwrite(pj(_root,"4.png"), img_step4)

    for res in Blocks["normal_blocks"]:
        x0,x1,y0,y1 = res[0][0],res[0][1],res[1][0],res[1][1]
        box = np.int0([[y0,x0], [y0,x1], [y1,x1], [y1,x0]])
        # print(res,box)
        cv2.drawContours(img_step3, [box], -1, (0,0,255),5)
    cv2.imwrite(pj(_root,"4.png"), img_step3)
    cv2.imwrite(pj(_root,"result.png"), img)
    # cv2.imwrite(r"D:\Projects\2021\qgis\blockShow\blockShow%d_%d.png"%(size,step), img)
    # cv2.imwrite(r"D:\Projects\2021\qgis\blockShow\blockShowArea%d_%d.png"%(size,step), img2)
# block_show()


def block_maks(size=64, step=8):
    img = cv2.imread(r"D:\Data\Forestry\QMNP\2010\combined\742.png")
    img2 = np.zeros(img.shape)
    f=open(r"D:\Projects\2021\qgis\Blocks_%d_%d"%(size, step),"rb")
    Blocks = pickle.load(f)
    f.close()

    

HSV_RED = [
    [[0, 36, 41], [10, 255, 255]],
    [[152,36,41], [182,255,255]]
]

def img_hsv_range(img, color_range=HSV_RED):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    masks = np.zeros((hsv.shape[0], hsv.shape[1]))

    for rg in color_range:
        # print(cv2.inRange(hsv, rg[0], rg[1]).dtype)
        masks += cv2.inRange(hsv, np.array(rg[0]), np.array(rg[1]))
        # 

    masks = np.where(masks != 0, 1, 0).astype(np.uint8)
    res = cv2.bitwise_and(img,img,mask = masks)
    # print(res.shape)
    return res


    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    # Min max stretch
    min_res = 0  # res.min()
    max_res = 255 # res.max()
    res = (255.0*(res - min_res)/(max_res - min_res)).astype(np.uint8)

    # cv2.imshow('res',res)
    # cv2.waitKey()
    return res


def img_combine(rgb="543"):
    """
    Combine band images to BGR format

    Params:
    Input: band order of RGB
    Outpu: Combined image of BGR
    """
    for _,fds,_ in os.walk(DATA_ROOT):
        break
    res = {}
    for fd in fds:
        _root = pj(DATA_ROOT,fd)
        r_f = pj(_root, "B%s.TIF"%rgb[0])
        g_f = pj(_root, "B%s.TIF"%rgb[1])
        b_f = pj(_root, "B%s.TIF"%rgb[2])
        

        b=cv2.imread(b_f, cv2.IMREAD_GRAYSCALE)
        g=cv2.imread(g_f, cv2.IMREAD_GRAYSCALE)
        r=cv2.imread(r_f, cv2.IMREAD_GRAYSCALE)
        # print(b.shape,r.shape,g.shape)
        new_bgr = cv2.merge((b,g,r))
        res[fd]=new_bgr
        # cv2.imshow("123",new_bgr)
        # cv2.waitKey(0)
        # exit()

    return res

def img_forest(rgb="543"):
    res = img_combine(rgb)
    mask_p = r"D:\Data\Forestry\QMNP\MASK_ROI.png"
    mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
    # print(mask.shape)
    for year, img in res.items():
        _root = pj(DATA_ROOT,year,"combined")
        if not os.path.exists(_root):
            os.mkdir(_root)

        img = img_hsv_range(img)
        img = cv2.bitwise_and(img,img,mask =mask)

        cv2.imwrite(pj(_root,"%s.png"%rgb), img)
 
# img_combine("742")

rgb="432"
img_forest(rgb)
# for _, fds, _ in os.walk(DATA_ROOT):
#     break  
# for fd in fds:
#     print(fd)
#     img = cv2.imread(pj(DATA_ROOT, fd , "combined/%s.png"%rgb)) # , cv2.IMREAD_GRAYSCALE
#     cv2.imshow("2", cv2.resize(img, (2000,400)))
#     cv2.waitKey(1000)




        