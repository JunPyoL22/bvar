import numpy as np
import pandas as pd
import os
import sys

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
NITER = 50
NSAVE = 25
HORIZON =10
NIND = data.shape[1]
NFACTOR = 3

os.chdir(MODULE_PATH)
from model import FactorAugumentedVARX, GFEVarianceDecompose, VarianceDecompositionMatrix
favarx = FactorAugumentedVARX(n_iter=NITER, n_save=NSAVE, lag=1,
                              var_lag=1, n_factor=NFACTOR, horizon=HORIZON,
                              smoother_option='CarterKohn',
                              is_standardize=False).estimate(data, z, w)

# average over the number of drawed
impulse_response = np.mean(favarx.impulse_response, axis=0)
et = np.mean(favarx.et, axis=0)
var_covar = np.dot(et, et.T)

CONTRI_RATE = GFEVarianceDecompose(HORIZON, impulse_response,
                                   var_covar).compute(NIND, NIND+NFACTOR).contri_rate

vdm = VarianceDecompositionMatrix(NIND, CONTRI_RATE[HORIZON]).calculate_spillover_effect()
spil_to_oths = vdm.spillover_to_oths
spil_from_oths = vdm.spillover_from_oths
net_effect = vdm.net_spillover