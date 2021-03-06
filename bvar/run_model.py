#%%
import os
import sys

import numpy as np
import pandas as pd
from bvar.filter import HpFilter
from db.monthly import LaborProductivity
from bvar.model import FactorAugumentedVARX, GFEVarianceDecompose, VarianceDecompositionMatrix
from bvar.utils import ExcelExporter

def get_productivty_variation(data, cycle_span):
    hp_filter = HpFilter(period_type=cycle_span)
    prd_cycle = np.empty(data.shape, dtype=np.float32)
    ind_codes = list(data.columns)
    ln_data = np.log(np.array(data, dtype=np.float32))
    for i, _ in enumerate(ind_codes):
        hp_filter.filtering(ln_data[:,i:i+1])
        prd_cycle[:, i:i + 1] = hp_filter.cycle
    return pd.DataFrame(prd_cycle, columns=ind_codes, index=data.index)

def calculate_industrial_linkage_data(weight, prd):
    total_rows, total_inds = prd.shape
    prd_star = np.empty((total_rows, total_inds), dtype=np.float32)
    for i in range(total_inds):
        temp = prd.copy()
        temp[:, i:i+1] = np.zeros((total_rows,1))
        prd_star[:, i:i+1] = np.dot(temp, weight.T[:, i:i+1])
    return prd_star

if __name__=="__main__":

        np.set_printoptions(precision=3, suppress=True)
        if sys.platform == 'win32':
            SYS = 'Window'
        else:
            SYS = 'Mac'

        if SYS is 'Mac':
            DATA_PATH = '/Users/Junpyo/Google Drive/data'
            MODULE_PATH = '/Users/Junpyo/project/bvar/bvar'

        if SYS is 'Window':
            DATA_PATH = 'D:\\Google Drive\\data'
            MODULE_PATH = 'C:\\project\\bvar\\bvar'

        #Import Input, Output data from DataBase and calculate monthly productivity
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
        prd_cycle = get_productivty_variation(pivoted_data, 'Month')
        new_column_names = ['C'+name if len(name)==2 else name for name in prd_cycle.columns]
        prd_cycle.columns = new_column_names
        prd_cycle.sort_index(axis=1, inplace=True)

        # Y_old = pd.read_csv('detrended_prd_index.csv', delimiter=',',
        #                 dtype=float, header=0, usecols=range(1,40))
        # new_names = np.sort(['C'+name for name in Y.columns if len(name)==2] + \
        #                     [name for name in Y.columns if len(name)==1])
        # Y_old.columns = new_names
        # Y_old = Y_old.ix[:, new_names]
        # X_star = pd.read_csv('Xstar.csv', delimiter=',',
        #                      dtype=np.float, header=0, usecols=range(2,41))

        # Import data from other external source like a excel file
        os.chdir(DATA_PATH)
        W_df = pd.read_csv('weight.csv', delimiter=',',
                            dtype=np.float32, header=0, usecols=range(2,41))

        prd = np.array(prd_cycle, dtype=np.float32)
        weight = np.array(W_df, dtype=np.float64)
        prd_star = calculate_industrial_linkage_data(weight, prd)

        # Set model params
        N_ITER = 100
        N_SAVE = 50
        HORIZON_1 = 5
        HORIZON_2 = 20
        N_IND = prd.shape[1] #39
        N_FACTOR = 3

        os.chdir(MODULE_PATH)
        favarx_1 = FactorAugumentedVARX(n_iter=N_ITER, n_save=N_SAVE, lag=1,
                                      var_lag=1, n_factor=N_FACTOR, horizon=HORIZON_1,
                                      smoother_option='CarterKohn',
                                      is_standardize=False).estimate(prd, prd_star, weight)

        # Average over the number of drawed
        imp_res_1 = np.mean(favarx_1.impulse_response, axis=0)
        et = np.mean(favarx_1.et, axis=0)
        var_covar_1 = np.dot(et.T, et)

        CONTRI_RATE_1 = GFEVarianceDecompose(HORIZON_1, imp_res_1,
                                           var_covar_1).compute(N_IND, N_IND+N_FACTOR).contri_rate

        vdm = VarianceDecompositionMatrix(N_IND, CONTRI_RATE_1[HORIZON_1]).calculate_spillover_effect()
        spil_to_oths = vdm.spillover_to_oths
        spil_from_oths = vdm.spillover_from_oths
        net_effect = vdm.net_spillover

        # Adjustment of contribution rate
        column_sum_contri_rate_1 = np.sum(CONTRI_RATE_1[HORIZON_1], keepdims=True, axis=1)
        adj_contri_rate_1 = 100*np.divide(CONTRI_RATE_1[HORIZON_1], column_sum_contri_rate_1)

        # Export result to excel file
        column_label = list(W_df.columns) + ['Macro_factor_1', 'Macro_factor_2', 'Macro_factor_3']
        result_1 = pd.DataFrame(adj_contri_rate_1,
                              columns=column_label)
        ExcelExporter(DATA_PATH, 'industrial_spillover_result_1.xlsx', result_1).write_data(sheetname='Sheet1')