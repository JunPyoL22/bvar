#%%
import os
import sys

import numpy as np
import pandas as pd
from bvar.filter import HpFilter
from db.monthly import LaborProductivity
from bvar.model import FactorAugumentedVARX, GFEVarianceDecompose, VarianceDecompositionMatrix

def get_productivty_variation(data, cycle_span):
    hp_filter = HpFilter(period_type=cycle_span)
    prd_cycle = np.empty(data.shape, dtype=np.float32)
    ind_codes = list(data.columns)
    for i, _ in enumerate(ind_codes):
        hp_filter.filtering(np.array(data)[:,i:i+1])
        prd_cycle[:, i:i + 1] = hp_filter.cycle

    return pd.DataFrame(prd_cycle, columns=ind_codes, index=data.index)

def calculate_industrial_likage_dataset(W, prd):
    total_rows, total_inds = prd.shape
    result = np.empty((total_rows, total_inds), dtype=np.float32)
    for j in total_inds:
        W[j:j+1, j:j+1] = 0
        for i in total_rows:
             result[i:i+1, j:j+1] = np.dot(prd, W.T)
    return result

if __name__=="__main__":

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

        # Input, Output from DataBase and calculate monthly productivity
        INDCODE_1DIGIT = ['B', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U']
        INDCODE_2DIGIT = [str(i) for i in range(10, 33)]
        INDCODE = INDCODE_1DIGIT + INDCODE_2DIGIT
        MONTH = [str(i) for i in range(1, 13)]
        START_YEAR = 2008
        LAST_YEAR = 2017

        lpd = LaborProductivity('prod', 'mh_input_t', INDCODE, START_YEAR, LAST_YEAR)
        monthly_prd = lpd.monthly_producvitiy.dropna()
        monthly_prd.loc[:,['year','mq']] = monthly_prd[['year','mq']].astype(np.int)
        monthly_prd.sort_values(by=['year', 'mq'], inplace=True)

        pivoted_data = pd.pivot_table(monthly_prd, values='prd_index', columns='kisc_code', index=['year','mq'])
        # Applying HP filter to monthly_prd
        Y = get_productivty_variation(pivoted_data, 'Month')
        new_column_names = ['C'+name if len(name)==2 else name for name in Y.columns]
        Y.columns = new_column_names
        Y.sort_index(axis=1, inplace=True)

        # import DATA from other external sources
        os.chdir(DATA_PATH)
        # Y_old = pd.read_csv('detrended_prd_index.csv', delimiter=',',
        #                 dtype=float, header=0, usecols=range(1,40))
        # new_names = np.sort(['C'+name for name in Y.columns if len(name)==2] + \
        #                     [name for name in Y.columns if len(name)==1])
        # Y_old.columns = new_names
        # Y_old = Y_old.ix[:, new_names]

        X_star = pd.read_csv('Xstar.csv', delimiter=',',
                             dtype=np.float, header=0, usecols=range(2,41))
        W = pd.read_csv('weight.csv', delimiter=',',
                        dtype=np.float, header=0, usecols=range(2,41))

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