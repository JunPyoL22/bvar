import numpy as np
import pandas as pd
import os
import sys

MAC = True
if sys.platform == 'win32':
    WIN = True

if MAC:
    DATA_PATH = '/Users/junpyolee/Google Drive/data'
    MODULE_PATH = '/Users/junpyolee/projects/bvar/bvar'
if WIN:
    DATA_PATH = 'D:\\Google Drive\\data'
    MODULE_PATH = 'C:\\project\\bvar\\bvar'

# DATA import
os.chdir(DATA_PATH)
Y = pd.read_csv('detrended_prd_index.csv', delimiter=',', 
                dtype=float, header=0, usecols=range(1,40))

new_names = np.sort(['C'+name for name in Y.columns if len(name)==2] + \
                    [name for name in Y.columns if len(name)==1])
Y.columns = new_names
Y = Y.ix[:, new_names]

X_star = pd.read_csv('Xstar.csv', delimiter=',', 
                     dtype=float, header=0, usecols=range(2,41))
W = pd.read_csv('weight.csv', delimiter=',', 
                dtype=float, header=0, usecols=range(2,41))

data = np.array(Y)
z = np.array(X_star)
w = np.array(W)

os.chdir(MODULE_PATH)
from model import FactorAugumentedVARX
favarx = FactorAugumentedVARX(n_iter=100, n_save=50, lag=1, var_lag=1, n_factor=3, 
                              smoother_option='CarterKohn', is_standardize=False).estimate(data, z, w)
