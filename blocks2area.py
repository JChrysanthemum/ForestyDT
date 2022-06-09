# Transform model's out-image to study-area
import cv2
import numpy as np
import pickle
from common import *
area_shape = (1146, 5470)

path_result = r'E:\Data\ForestDT_Result'
BAND="B3"
IMG_SIZE=128
IMG_STEP=16

# result use 500


def img_statics(size,step):
    """
    res[band][file]
    """
    f_name = "statics_ori%d_%d"%(size,step)
    if os.path.exists(f_name):
        f = open(f_name,'rb')
        res = pickle.load(f)
        f.close()
        return res

    _root = r'D:\Data\Forestry\QMNP_Block\%d_%d'%(size,step)
    years = list_folders(_root)
    bands = list_folders(pj(_root,years[0]))
    files = list_files(pj(_root,years[0],bands[0]))
    y_num = len(years)
    res = {}
    # print(years,bands,files)
    for band in bands:
        res[band]={}
        for f in files:
            # min max mean
            f_min, f_max, f_mean, f_std = 0,0,0,0
            # print(f,band)
            for y in years:
                img_path=pj(_root,y,band,f)
                img = cv2.imread(img_path)
                f_min, f_max, f_mean, f_std = f_min+img.min(), f_max+img.max(), f_mean+img.mean(), f_std+img.std()
                # print(y,img.min(), img.max(), img.mean(), img.std())
            f_min, f_max, f_mean, f_std = f_min/y_num, f_max/y_num, f_mean/y_num, f_std/y_num
            res[band][f] = [f_min, f_max, f_mean, f_std]
            # print(f_min, f_max, f_mean, f_std)
            # exit()
    f = open(f_name,'wb')
    pickle.dump(res, f)
    f.close()
    return res


def block_denormlize(_path = path_result):
    
    def img_hist_revert(img, img_max, img_min):
        img_norm = img*(img_max - img_min)/255.0 + img_min
        img_norm = img_norm.astype(np.uint8)
        return img_norm
    
    # models = list_folders(_path)
    
    models = ['LSTM_CGAN', 'LSTM_Conv', 'LSTM_WGAN', 'VAE_CGAN', 'VAE_WGAN']
    bands = ["B2","B3","B4"]

    for m in models:
        for band in bands:
            statics = img_statics(IMG_SIZE, IMG_STEP)[band]
            path_block = pj(_path,m,band)
            
            path_denorm = pj(_path,m,"denorm",band)
            mkdir2(path_denorm)
            
            files = list_files(path_block)
            for f in files:
                [f_min, f_max, f_mean, f_std] = statics[f]
                img = cv2.imread(pj(path_block,f), cv2.IMREAD_GRAYSCALE)
                img = img_hist_revert(img, f_max, f_min)
                cv2.imwrite(pj(path_denorm,f), img)
    
        

def blocks2area(_path = path_result):
    
    def block_weight(size=64, step=0):
        mask = np.zeros(area_shape,dtype=np.float)
        f=open(r"D:\Projects\2022\ForestyDT\Blocks_%d_%d"%(size, step),"rb")
        Blocks = pickle.load(f)
        f.close()

        h,w = area_shape[:2]
        locs = Blocks["normal_blocks"]+ Blocks["patch_blocks"]
        for res in locs:
            x0,x1,y0,y1 = res[0][0],res[0][1],res[1][0],res[1][1]
            mask[x0:x1,y0:y1] +=1
        mask = 1.0/mask
        # f=open(r"D:\Projects\2021\qgis\%d-mask"%size,"wb")
        # pickle.dump(mask, f)
        # f.close()
        return mask,locs

    def merge_locs(mask,locs,_path):
        # cv2.imshow("mask",(255-mask*255).astype(np.uint8))

        back = np.zeros(area_shape,dtype=np.float)
        imgs = list_files(_path)
        for img in imgs:
            idx = int(img.split(".")[0])
            cliped = cv2.imread(pj(_path,img),cv2.IMREAD_GRAYSCALE)
            res = locs[idx]
            x0,x1,y0,y1 = res[0][0],res[0][1],res[1][0],res[1][1]
            back[x0:x1,y0:y1] +=cliped
            # org=((y0+y1)//2,(x0+x1)//2)
            # back = cv2.putText(back, str(idx), org, cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2, cv2.LINE_AA)
            # break

        back= (back*mask).astype(np.uint8)
        # cv2.imshow("e",back)
        # cv2.waitKey(0)
        return back

    # models = list_folders(_path)
    models = ['LSTM_CGAN', 'LSTM_Conv', 'LSTM_WGAN', 'VAE_CGAN', 'VAE_WGAN']
    
    mask,locs = block_weight(IMG_SIZE,IMG_STEP)
    roi_mask = cv2.imread(r"D:\Data\Forestry\QMNP\MASK_ROI.png", cv2.IMREAD_GRAYSCALE)
    
    bands = ["B2","B3","B4"]

    for m in models:
        for band in bands:
            path_denorm = pj(_path,m,"denorm",band)
            # path_denorm = pj(_path,m,band)
            path_area = pj(_path,m)
            img = merge_locs(mask, locs, path_denorm)
            img = cv2.bitwise_and(img,img,mask =roi_mask)
            cv2.imwrite(pj(path_area,"%s.png"%band), img)
         
         
def merge_area(_path = path_result, rgb="432"):
    models = list_folders(_path)
    # models = ['LSTM_CGAN', 'LSTM_Conv', 'LSTM_WGAN', 'VAE_CGAN', 'VAE_WGAN']


    for m in models:
        path_area = pj(_path,m)
        for f in list_files(path_area):
            if f.find('B')>-1 :
                suffix = f.split(".")[-1]
                break
        r_f = pj(path_area, "B%s.%s"%(rgb[0],suffix))
        g_f = pj(path_area, "B%s.%s"%(rgb[1],suffix))
        b_f = pj(path_area, "B%s.%s"%(rgb[2],suffix))


        b=cv2.imread(b_f, cv2.IMREAD_GRAYSCALE)
        g=cv2.imread(g_f, cv2.IMREAD_GRAYSCALE)
        r=cv2.imread(r_f, cv2.IMREAD_GRAYSCALE)
        # print(b.shape,r.shape,g.shape)
        new_bgr = cv2.merge((b,g,r))

        mask_p = r"D:\Data\Forestry\QMNP\MASK_ROI.png"
        mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
        new_bgr = cv2.bitwise_and(new_bgr,new_bgr,mask =mask)
        cv2.imwrite(pj(path_area,"%s.png"%rgb), new_bgr)
      
            
def red_segment(_path = path_result):
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
        return masks*255
        
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

    models = list_folders(_path)
    # models = ['LSTM_CGAN', 'LSTM_Conv', 'LSTM_WGAN', 'VAE_CGAN', 'VAE_WGAN']


    for m in models:
        path_area = pj(_path,m)
        tar_img = cv2.imread(pj(path_area,"432.png"))
        seg_img = img_hsv_range(tar_img)
        cv2.imwrite(pj(path_area,"segment.png"), seg_img)



# output2blocks()


# Step 1. Denormlize block image to their distribution
# block_denormlize()

# Step 2. Gather blocks to study area
# blocks2area()

# Step 3. Merge area to rgb
# merge_area()

# Step 4. Segment from 432 band image
# red_segment()