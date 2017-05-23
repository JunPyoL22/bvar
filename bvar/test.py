import numpy as np
import pandas as pd
import os
import sys

def get_average_value(ir, var_covar, nsave)
    sum_ir = 0
    sum_var_covar = 0
    for i in range(nsave):
        sum_ir = sum_ir + ir[i]
        sum_var_covar = sum_var_covar + var_covar[i]
    return np.divide(sum_ir, nsave), np.divide(sum_var_covar, nsave)

np.set_printoptions(precision=3, suppress=True)
if sys.platform == 'win32':
    SYS = 'WIN'
else:
    SYS = 'MAC'

if SYS is 'MAC':
    DATA_PATH = '/Users/Junpyo/Google Drive/data'
    MODULE_PATH = '/Users/Junpyo/project/bvar/bvar'
if SYS is 'WIN':
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

data = np.array(Y, dtype=float)
z = np.array(X_star, dtype=float)
w = np.array(W, dtype=float)

# model constant
NITER = 100
NSAVE = 50
HORIZON =10
NIND = data.shape[1]
NFACTOR = 3

os.chdir(MODULE_PATH)
from model import FactorAugumentedVARX, GFEVarianceDecompose
favarx = FactorAugumentedVARX(n_iter=NITER, n_save=NSAVE, lag=1,
                              var_lag=1, n_factor=NFACTOR, horizon=HORIZON,
                              smoother_option='CarterKohn',
                              is_standardize=False).estimate(data, z, w)

avg_impulse_response, \
avg_var_covar = get_average_value(favarx.impulse_response,
                                  favarx.var_covar,
                                  NSAVE)

CONTRI_RATE = GFEVarianceDecompose(HORIZON, avg_impulse_response,
                                   avg_var_covar).compute(NIND, NIND+NFACTOR).contri_rate

