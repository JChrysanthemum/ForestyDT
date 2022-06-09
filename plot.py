import pandas as pd
import pickle
import matplotlib
from matplotlib import pyplot as plt
from common import *
from blocks2area import IMG_SIZE, IMG_STEP, path_result
from blocks2area import img_statics
from evaluate import path_scores
import numpy as np

models = ['LSTM_CGAN', 'LSTM_Conv', 'LSTM_WGAN', 'VAE_CGAN', 'VAE_WGAN']
bands = ["B2","B3","B4"]



# for f in files:
#     res = []
#     for band in bands:
#         res.append(statics[band][f][-1])
#     print(res)

def data_curves():
    statics = img_statics(IMG_SIZE, IMG_STEP)

    
    files = list_files(pj(path_result,models[0],bands[0]))
    
    res_stdev={}
    for band in bands:
        res_stdev[band] = []
        for f in files:
            res_stdev[band].append(statics[band][f][-1])

    f=open(pj(path_scores,"blocks_score"),"rb")
    res_nrmse = pickle.load(f)
    f.close()


    bands = ["B2","B3","B4"]
    for band in bands:
        for stdev, score in zip(res_stdev[band], res_nrmse["LSTM_CGAN"][band]):
            print(stdev,score)

    # save_pickle(pj(path_scores,"res_stdev"), res_stdev)
    # save_pickle(pj(path_scores,"res_nrmse"), res_nrmse)


def plot_scats():
    plt.rcParams["font.size"] = 15
    res_cr=load_pickle(pj(path_scores,"b432_score"))
    res_nrmse=load_pickle(pj(path_scores,"blocks_score"))
    sum_nrmse={}
    for model in models:
        scores = []
        for band in bands:
            scores.append(np.mean(res_nrmse[model][band]))
        sum_nrmse[model] = np.mean(scores)
        
    x = [sum_nrmse[model] for model in models]
    y = [res_cr[model] for model in models]
    
    
    
    
    fig, ax = plt.subplots(dpi=300)
    ax.scatter(x, y, s = 75,color=["#67b9f5","#7be397","#f9cf81","#fda193","#7fdbd4"] , alpha = 0.8) # #9c8ad2
    ax.invert_xaxis()
    ax.set_xlabel('NRMSE')
    ax.set_ylabel('CR')
    # for i, txt in enumerate(models):
    #     ax.annotate(txt, (x[i], y[i]))
    
    plt.grid(linestyle='-.')
    plt.savefig(pj(path_scores,"scatter.png"), bbox_inches='tight', transparent=True)
    plt.clf()
    

def plot_3dscats():
    color=["#67b9f5","#7be397","#f9cf81","#fda193","#7fdbd4"]
    res_nrmse=load_pickle(pj(path_scores,"blocks_score"))
    x=[]
    y=[]
    z=[]
    
    for model in models:
        x.append(np.mean(res_nrmse[model]["B2"]))
        y.append(np.mean(res_nrmse[model]["B3"]))
        z.append(np.mean(res_nrmse[model]["B4"]))
          
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x, y, z,color=color , alpha = 0.8)
    # ax.invert_yaxis()
    ax.invert_xaxis()
    ax.set_xlabel('B2')
    ax.set_ylabel('B3')
    ax.set_zlabel('B4')
    

    
    
    # ax.scatter(x, z,marker = "x",color=color , alpha = 0.8)
    # for i in range(1):
    #     for _x,_y,_z in [[x[i],y[i],zmin], [x[i],ymin,z[i]], [xmin,y[i],z[i]]]:
    #         print([[x[i],y[i],zmin], [x[i],ymin,z[i]], [xmin,y[i],z[i]]])
    #         ax.scatter(_x,_y,_z, marker = "x",color=color[i] , alpha = 0.8)
    
    # ax.plot(x, z,'r+', zdir='y', zs = 0.425, color=color)
    
    # ax.plot(y, z, 'g+', zdir='x', zs = 0.45)
    # ax.plot(x, y, 'k+', zdir='z', zs = 0.55)
    
    # for i, txt in enumerate(models):
    #     ax.annotate(txt, (x[i], y[i], z[i]))
    
    plt.grid(linestyle='-.')
    plt.savefig(pj(path_scores,"3d scatter.png"), bbox_inches='tight', transparent=True)
    plt.clf()


def data_tables():
    res_nrmse = load_pickle(pj(path_scores,"blocks_score"))
    cr = load_pickle(pj(path_scores,"b432_score"))
    res = []
    
    for model in models:
        bs = [res_nrmse[model][band] for band in bands]
        bs = [[np.mean(a), np.std(a)] for a in bs]
        bs_str =[]
        for b in bs:
            bs_str.append("(%.2f,rep %.2f) "%(b[0], b[1]))
        # print("%s %s %.2f"%(model,bs_str, cr[model]))
        res.append([model, *bs_str, "%.2f"%cr[model]])
    res = pd.DataFrame(res)
    res.to_csv(pj(path_scores,"values.csv"))
    print(res)


# data_curves()

# plot_scats()

# plot_3dscats()

data_tables()