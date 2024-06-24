from methods.fair_gumbel_one import *
from methods.tools import pooled_least_squares
import numpy as np
import os
import argparse
import time
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso,LogisticRegression

def acc(traindata,testdata):
    n = len(traindata)
    num = 0
    for i in range(n):
          if(abs(traindata[i]-testdata[i])<0.5):
             num=num+1
    print(num/n)

r_water_list=[0.95,0.75,0.5]
r_land_list=[0.9,0.7,0.5]
x=[]
y=[]
z=[]

for i in range(len(r_water_list)):
    pca = PCA(n_components=500)
    r_water=r_water_list[i]
    r_land=r_land_list[i]
    x_org=np.load(f'./res/train/rwater_{r_water}_rland_{r_land}_x.npy')
    pincipal_componets = pca.fit_transform(x_org)
    x.append(pincipal_componets)
    #x.append(x_org)
    y.append(np.load(f'./res/train/rwater_{r_water}_rland_{r_land}_y.npy'))
    z.append(np.load(f'./res/train/rwater_{r_water}_rland_{r_land}_z.npy'))

x_test=np.load(f'./res/test/rwater_0.05_rland_0.05_x.npy')
y_test=np.load(f'./res/test/rwater_0.05_rland_0.05_y.npy')
z_test=np.load(f'./res/test/rwater_0.05_rland_0.05_z.npy')
pca = PCA(n_components=500)
pincipal_componets_test = pca.fit_transform(x_test)

lassoy = Lasso(alpha=0.001,max_iter=1000)
catx=np.concatenate((x[0],x[1]))
caty=np.concatenate((y[0],y[1]))
lassoy.fit(x[2],y[2])
y0=lassoy.predict(pincipal_componets_test)
#y0=lassoy.predict(x_test)
acc(y0,y_test)
y0=lassoy.predict(x[2])
acc(y0,y[2])
print(np.where(lassoy.coef_>1e-10))
