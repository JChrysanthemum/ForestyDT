import os 
import cv2
from common import *
from common import _list_files
import pickle
import numpy as np


src_root = r'D:\Data\Forestry\QMNP'
trg_root = r'D:\Data\Forestry\QMNP_Block'
config_dict= {
    64:{8:r'D:\Projects\2021\qgis\Blocks_64_8'},
    128:{16:r'D:\Projects\2021\qgis\Blocks_128_16',
        0:r'D:\Projects\2021\qgis\Blocks_128_0'},
    256:r'D:\Projects\2021\qgis\Blocks_256_32'
}


def create(size=64, step = 8,renew = False):
    f=open(config_dict[size][step],"rb")
    config = pickle.load(f)
    f.close()
    locs=config['normal_blocks'] + config['patch_blocks']

    # print( config['patch_blocks'])
    # exit()
    years = [str(i) for i in range(2001,2022)]
    bands = ["B%d"%i for i in range(1,9)]
    # print(years, bands)

    new_root = pj(trg_root,"%d_%d"%(size,step))
    mkdir2(new_root, renew)
    for year in years:
        mkdir2(pj(new_root,year), renew)
        print("Year %s starting"%year)
        for band in bands:
            sav_root = pj(new_root,year,band)
            mkdir2(sav_root, renew)
            img = cv2.imread(pj(src_root,year,"%s.TIF"%band), cv2.IMREAD_GRAYSCALE)# todo gray
            print("Band %s"%band)
            for i in range(len(locs)):
                w_i = locs[i][1]
                h_i = locs[i][0]
                # print(w_i,h_i)
                # exit()
                # Debug modeL Visualize windows moving
                # cp_img = cv2.imread(pj(src_root,year,"%s.TIF"%band))
                # mask = cv2.imread(r'D:\Data\Forestry\QMNP\MASK_ROI.png', cv2.IMREAD_GRAYSCALE)
                # cp_img = cv2.bitwise_and(cp_img, cp_img, mask=mask)
                # box = np.array([[w_i[0],h_i[0]],[w_i[0],h_i[1]],[w_i[1],h_i[1]],[w_i[1],h_i[0]]])
                # cv2.drawContours(cp_img,[box], -1, (0,0,255), 10)
                # cv2.imshow("data", cv2.resize(cp_img, (2000,400)))
                # cv2.waitKey(1)

                # print(i)
                clip = img[h_i[0]:h_i[1], w_i[0]:w_i[1]]
                if clip.sum() == 0 :
                    print('BUG')
                cv2.imwrite(pj(sav_root,"%04d.png")%i, clip)
                # print(clip.shape)
                # exit()
                # if img[h_i[0]:h_i[1], w_i[0]:w_i[1]].sum() == 0:
                #     print("?")
            # exit()
            # print(img.shape, pj(src_root,year,"%s.TIF"%band))

    # print('%04d'%1)
    # for i in range(len(locs)):
    #     w_i = locs[i][0]
    #     h_i = locs[i][0]
    #     print()
    
# create(size=64)
create(size=128,step=0,renew=True)
create(size=128,step=16,renew=True)
# create(size=256)
