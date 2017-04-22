DATA_PATH = '/Users/junpyolee/Google Drive/data'
MODULE_PATH = '/Users/junpyolee/projects/bvar/bvar'

import numpy as np
import pandas as pd
import os

# DATA import
os.chdir(DATA_PATH)
Y = pd.read_csv('detrended_prd_index.csv', delimiter=',', 
                dtype=float, header=0, usecols=range(1,40))
X_star = pd.read_csv('Xstar.csv', delimiter=',', 
                     dtype=float, header=0, usecols=range(2,41))
W = pd.read_csv('weight.csv', delimiter=',', 
                dtype=float, header=0, usecols=range(2,41))

os.chdir(MODULE_PATH)
from model import FactorAugumentedVARX
favarx = FactorAugumentedVARX(n_iter=100, n_save=50, lag=1, var_lag=1, n_factor=3,
                              alpha0=None, V0=None, V0_scale=1, v0=None, S0=None, 
                              smoother_option='DurbinKoopman', is_standardize=False)
