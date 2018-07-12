# if os.getcwd()!='C:\\project\\Quaterly_labor_productivity_index':
#     os.chdir('C:\\project\\Quaterly_labor_productivity_index')
import numpy as np
import pandas as pd

from db.database import db_session
from db.extract import LpsExtractor


class DbData:
    db = LpsExtractor(db_session)
    print('LPS DB connected')
    def __init__(self, type, time, kisc_code,
                 start_year, last_year):
        self._type = type
        self._time = time
        self._kisc_code = kisc_code
        self._start_year = start_year
        self._last_year = last_year
        self._data = None
        self._db = self.db

    @property
    def data(self):
        self._preprocess()
        return self._data

    def _preprocess(self):
        raise NotImplementedError

class LaborProductivity:
    def __init__(self, output_type, input_type, kisc_code,
                    start_year, last_year):
        '''
        :param output_type: str, Must be one of these 'va', 'prod'
        :param input_type: str,  Must be one of these mh_input, mm_input, mh_input_t
        '''
        self._outputType = output_type
        self._inputType = input_type
        self._kisc_code = kisc_code
        self._start_year = start_year
        self._last_year = last_year

    @property
    def monthly_producvitiy(self):
        assert self._outputType is not 'va' \
            ,'Value_added based Productivity cannot be monthly' \
             'this is only for quaterly'
        return MonthlyProductivity(self._outputType,
                                   self._inputType,
                                   self._kisc_code,
                                   self._start_year,
                                   self._last_year).calculate_productivity()

class MonthlyProductivity:
    month_list = [str(i) for i in range(1,13)]

    def __init__(self, output_type, input_type,
                 kisc_code, start_year, last_year):
        self._data_set = None
        self._output = OutputData(output_type, self.month_list,
                                  kisc_code, start_year, last_year).data
        self._input = InputData(input_type, self.month_list,
                                kisc_code, start_year, last_year).data


    def _preprocess(self):
        self._output.loc[:, 'ip_index'] = self._output.loc[:, 'ip_index']\
                                                      .astype(np.float64)
        self._data_set = self._merge(self._output, self._input)
        return self
        
    def calculate_productivity(self):
        self._preprocess()
        self._data_set.loc[:, 'prd_index'] = \
            100*self._data_set['ip_index']/self._data_set['mh_input']
        return self._data_set.drop(self._data_set.columns[[3, 4, 5]], axis=1)

    def _merge(self, data, base_data):
        return pd.merge(data, base_data, on=['year','mq','kisc_code'], how='left')

class OutputData(DbData):
    def __init__(self, type, time, kisc_code,
                 start_year, last_year):
        super(OutputData, self).__init__(type, time, kisc_code,
                                         start_year, last_year)

    def _preprocess(self):
        data = self._db.get_output_index(self._type,
                                         kisc_code=self._kisc_code,
                                         start_year=self._start_year,
                                         last_year=self._last_year)
        self._data = data[data.mq.isin(self._time)]

class InputData(DbData):
    def __init__(self, type, time, kisc_code,
                 start_year, last_year):
        super(InputData, self).__init__(type, time, kisc_code,
                                        start_year, last_year)
    
    @property
    def data(self):
        self._preprocess(base_year=2010)
        return self._data

    def _preprocess(self, base_year=None):
        var_name = self._type
        input_data_level = self._get_laborinput_lavel()
        base = BaseDataProvider(base_year).get_base_data(
                                                input_data_level,
                                                var_name,
                                                level=['kisc_code'])
        data = self._merge(input_data_level, base)
        self._data = self.calculate_monthly_index(data)
        self._data.drop(self._data.columns[[3, 4]], axis=1, inplace=True)
        return self

    def _get_laborinput_lavel(self):
        return InputLevelData(self._type, self._time, self._kisc_code,
                              self._start_year, self._last_year).data

    def _merge(self, data, base_data):
        return pd.merge(data, base_data, on='kisc_code', how='left')

    def calculate_monthly_index(self, data_set):
        data_set.loc[:, 'mh_input_t'] = data_set.loc[:,'mh_input_t'].astype(np.float64)
        data_set['mh_input'] = 100*data_set['mh_input_t']/data_set['mh_input_t_2010']
        return data_set


class InputLevelData(DbData):
    def __init__(self, type, time, kisc_code,
                 start_year, last_year):
        super(InputLevelData, self).__init__(type, time, kisc_code,
                                             start_year, last_year)

    @property
    def data(self):
        self._preprocess()
        return self._level_data

    def _preprocess(self):
        level_data = self._db.get_laborinput_level(self._type,
                                                   kisc_code=self._kisc_code,
                                                   start_year=self._start_year,
                                                   last_year=self._last_year)
        self._level_data = level_data[level_data.mq.isin(self._time)]


class BaseDataProvider:
    def __init__(self, year):
        self._year = year
        self._base = None

    def get_base_data(self, data, var_name, level=None):
        self._preprocess_base(data, var_name, level)
        var_names = list(self._base.columns)
        base_mean = self._calculate_mean_data(var_names, level)
        return base_mean

    def _calculate_mean_data(self, var_names, level):
        return self._base[var_names].groupby(level, as_index=False)\
                                    .mean()

    def _preprocess_base(self, data, var_name, level):
        selected_data = self._select_data(data)
        new_varName = '_'.join([var_name, '2010'])
        selected_data.loc[:, new_varName] = selected_data.loc[:, var_name]\
                                                         .astype(np.float64)
        selected_data.drop([var_name], axis=1, inplace=True)
        self._base = selected_data
        return self

    def _select_data(self, data):
        return data.loc[data['year']==str(self._year)].copy()

if __name__ == '__main__':

    INDCODE_1DIGIT = ['B', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U']
    INDCODE_2DIGIT = [str(i) for i in range(10, 33)]
    INDCODE = INDCODE_1DIGIT + INDCODE_2DIGIT
    MONTH = [str(i) for i in range(1,13)]
    START_YEAR = 2008
    LAST_YEAR = 2017

    lpd = LaborProductivity('prod', 'mh_input_t', INDCODE, START_YEAR, LAST_YEAR)
    monthly_prd = lpd.monthly_producvitiy

