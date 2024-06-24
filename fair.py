from methods.fair_gumbel_one import *
from methods.tools import pooled_least_squares
import numpy as np
import os
import argparse
import time
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso,LogisticRegression


r_water_list=[0.95,0.75]
r_land_list=[0.9,0.7]
x=[]
y=[]
z=[]
pca = PCA(n_components=500)
for i in range(len(r_water_list)):
    r_water=r_water_list[i]
    r_land=r_land_list[i]
    x_org=np.load(f'./res/train/rwater_{r_water}_rland_{r_land}_x.npy')
    #x_org=np.delete(x_org,tuple(d),axis=1)#删除LASSO选择的变量
    pincipal_componets = pca.fit_transform(x_org)
    x.append(pincipal_componets)
    y.append(np.load(f'./res/train/rwater_{r_water}_rland_{r_land}_y.npy'))
    z.append(np.load(f'./res/train/rwater_{r_water}_rland_{r_land}_z.npy'))
x_test=np.load(f'./res/test/rwater_0.05_rland_0.05_x.npy')
y_test=np.load(f'./res/test/rwater_0.05_rland_0.05_y.npy')
z_test=np.load(f'./res/test/rwater_0.05_rland_0.05_z.npy')
#x_test=np.delete(x_test,tuple(d),axis=1)
pincipal_componets1 = pca.fit_transform(x_test)


#packs = fairnn_sgd_gumbel_uni(x, y,(x_test,y_test),hyper_gamma=1, learning_rate=1e-4, niters=10000,log=True)
#packs = fairnn_sgd_gumbel_refit(x, y,mask,(x_test,y_test),log=True)
packs = fair_ll_classification_sgd_gumbel_uni(x, y,(pincipal_componets1,y_test),hyper_gamma=40, learning_rate=1e-2,niters=50000,log=True,)