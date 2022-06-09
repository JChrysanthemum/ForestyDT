import os
import cv2
from common import *
from os.path import join as pj
from sklearn.metrics import r2_score, mean_squared_error
from statistics import stdev
import pickle
import matplotlib.pyplot as plt 
from scipy.special import expit
from matplotlib.pyplot import figure
import numpy as np
import collections
from blocks2area import IMG_SIZE, IMG_STEP, path_result
from blocks2area import img_statics

path_scores = r'D:\Projects\2022\ForestyDT\result'

statics = img_statics(IMG_SIZE, IMG_STEP)
roi_mask = cv2.imread(r"D:\Data\Forestry\QMNP\MASK_ROI.png", cv2.IMREAD_GRAYSCALE)
area = (roi_mask!=0).sum()
back = (roi_mask==0).sum()

def NRMSE(img1, img2, band, name):
    img_stdev = statics[band][name][-1]
    score = np.sqrt(mean_squared_error(img1,img2))/img_stdev
    return score

def RMSE(img1, img2):

    score = np.sqrt(mean_squared_error(img1,img2))
    return score


def CR(img1, img2, tol=10):
    score = ((img1==img2).sum() - back)/area
    return score


def blocks_score(_path = path_scores):
    models = ['LSTM_CGAN', 'LSTM_Conv', 'LSTM_WGAN', 'VAE_CGAN', 'VAE_WGAN']
    bands = ["B2","B3","B4"]
    res = {}
    for m in models:
        res[m]={}
        for band in bands:
            res[m][band] = []
            path_denorm = pj(path_result,m,"denorm",band)
            path_true = pj(path_result,"True",band)
            files = list_files(path_denorm)
            files.sort()
            
            for f in files:
                img1 = cv2.imread(pj(path_denorm,f), cv2.IMREAD_GRAYSCALE)
                img2 = cv2.imread(pj(path_true,f), cv2.IMREAD_GRAYSCALE)
                res[m][band].append(NRMSE(img1, img2, band, f))

    f=open(pj(_path,"blocks_score"),"wb")
    pickle.dump(res, f)
    f.close()


def blocks_score_std(_path = path_scores):
    models = ['LSTM_CGAN', 'LSTM_Conv', 'LSTM_WGAN', 'VAE_CGAN', 'VAE_WGAN']
    bands = ["B2","B3","B4"]
    res = {}
    for m in models:
        res[m]={}
        for band in bands:
            res[m][band] = []
            path_denorm = pj(path_result,m,"denorm",band)
            path_true = pj(path_result,"True",band)
            files = list_files(path_denorm)
            files.sort()
            
            for f in files:
                img1 = cv2.imread(pj(path_denorm,f), cv2.IMREAD_GRAYSCALE)
                img2 = cv2.imread(pj(path_true,f), cv2.IMREAD_GRAYSCALE)
                res[m][band].append(RMSE(img1, img2))

    f=open(pj(_path,"blocks_score2"),"wb")
    pickle.dump(res, f)
    f.close()
    

    

def b432_score(_path = path_scores):
    models = ['LSTM_CGAN', 'LSTM_Conv', 'LSTM_WGAN', 'VAE_CGAN', 'VAE_WGAN']
    bands = ["B2","B3","B4"]
    res = {}
    tar = cv2.imread(pj(path_result,"True/segment.png"), cv2.IMREAD_GRAYSCALE)
    
    for m in models:
        img = cv2.imread(pj(path_result,m,"segment.png"), cv2.IMREAD_GRAYSCALE)
        res[m]=CR(tar,img)
    f=open(pj(_path,"b432_score"),"wb")
    pickle.dump(res, f)
    f.close()
            
# Evlauation 1: NRMSE of blocks
# blocks_score()
blocks_score_std()

# Evlauation 2: CR of b432 in red
# b432_score() 
    


