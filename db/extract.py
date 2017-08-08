
import pandas as pd
from sqlalchemy import and_ , outerjoin
from sqlalchemy.sql import select

from db.models import Lps003, Lps103, Lps141, Lps142, \
    Lps143, Lps144, Lps211, Lps212, \
    Lps221, Lps222, Lps223, Lps224, \
    Lps231, Lps233, Lps241, Lps243, \
    Lps251, Lps252, Lps253, Lps254


class LpsExtractor(object):
    def __init__(self, session):
        self.session = session

    def get_lpd_index(self, type='va', *, kisc_code=None,
                      start_year=None, last_year=None): 
        if type == 'va':
            data = pd.DataFrame(
                      self.session.query(Lps241.year, Lps241.mq, Lps241.kisc_code,
                                         Lps241.mh_add_index, Lps243.mh_add_rate)\
                               .filter(Lps241.year == Lps243.year)\
                               .filter(Lps241.mq == Lps243.mq)\
                               .filter(Lps241.kisc_code == Lps243.kisc_code)\
                               .filter(Lps241.year>=start_year)\
                               .filter(Lps241.year<=last_year)\
                               .all())
            return data.loc[data['kisc_code'].isin(kisc_code)]
        
        if type == 'prod':
            data = pd.DataFrame(
                    self.session.query(Lps231.year, Lps231.mq, Lps231.kisc_code,
                                      Lps231.mh_prod_index4, Lps233.mh_prod_rate4)\
                               .filter(Lps231.year == Lps233.year)\
                               .filter(Lps231.mq == Lps233.mq)\
                               .filter(Lps231.kisc_code == Lps233.kisc_code)\
                               .filter(Lps231.year>=start_year)\
                               .filter(Lps231.year<=last_year)\
                               .all())
            return data.loc[data['kisc_code'].isin(kisc_code)]
        
    def get_output_index(self, type='va', *, kisc_code=None,
                         start_year=None, last_year=None):
        if type == 'va':
            data = pd.DataFrame(
                      self.session.query(Lps142.year, Lps142.mq, Lps142.kisc_code,
                                         Lps142.cp_gva_index, Lps144.cp_gva_rate)\
                               .filter(Lps142.year == Lps144.year)\
                               .filter(Lps142.mq == Lps144.mq)\
                               .filter(Lps142.kisc_code == Lps144.kisc_code)\
                               .filter(Lps142.year>=start_year)\
                               .filter(Lps142.year<=last_year)\
                               .all())

            return data.loc[data['kisc_code'].isin(kisc_code)]
            
        if type == 'prod':
            if start_year > 2008:
                data = pd.DataFrame(
                          self.session.query(Lps003.year, Lps003.mq, Lps003.kisc_code,
                                             Lps003.ip_index, Lps103.ip_index_rate)\
                                   .filter(Lps003.year == Lps103.year)\
                                   .filter(Lps003.mq == Lps103.mq)\
                                   .filter(Lps003.kisc_code == Lps103.kisc_code)\
                                   .filter(Lps003.year>=start_year)\
                                   .filter(Lps003.year<=last_year)\
                                   .all())

                return data.loc[data['kisc_code'].isin(kisc_code)]
            else:
                join_query = outerjoin(Lps003, Lps103, onclause=and_(
                                                                 Lps003.year == Lps103.year,
                                                                 Lps003.mq == Lps103.mq,
                                                                 Lps003.kisc_code == Lps103.kisc_code
                                                                ))
                selected = select([Lps003.year, Lps003.mq, Lps003.kisc_code, Lps003.ip_index, Lps103.ip_index_rate])\
                                .where(Lps003.year != '0')\
                                .select_from(join_query)

                data = pd.DataFrame(self.session.execute(selected).fetchall(),
                                    columns=['year', 'mq', 'kisc_code', 'ip_index', 'ip_index_rate'])
                return data.loc[data['kisc_code'].isin(kisc_code)]

    def get_input_index(self, type='mh_input', *, kisc_code=None,
                        start_year=None, last_year=None):
        if type == 'mh_input':
            data = pd.DataFrame(
                      self.session.query(Lps221.year, Lps221.mq, Lps221.kisc_code,
                                         Lps221.mh_input_index4, Lps223.mh_input_rate4)\
                               .filter(Lps221.year == Lps223.year)\
                               .filter(Lps221.mq == Lps223.mq)\
                               .filter(Lps221.kisc_code == Lps223.kisc_code)\
                               .filter(Lps221.year>=start_year)\
                               .filter(Lps221.year<=last_year)\
                               .all())

            return data.loc[data['kisc_code'].isin(kisc_code)]
            
        if type == 'mm_input':
            data = pd.DataFrame(
                      self.session.query(Lps222.year, Lps222.mq, Lps222.kisc_code,
                                         Lps222.mm_input_index4, Lps224.mm_input_rate4)\
                               .filter(Lps222.year == Lps224.year)\
                               .filter(Lps222.mq == Lps224.mq)\
                               .filter(Lps222.kisc_code == Lps224.kisc_code)\
                               .filter(Lps222.year>=start_year)\
                               .filter(Lps222.year<=last_year)\
                               .all())
                               
            return data.loc[data['kisc_code'].isin(kisc_code)]

        if type == 'hh_input':
            return self.calculate_hh_input_index(kisc_code=kisc_code,
                                    start_year=start_year,
                                    last_year=last_year)


    def calculate_hh_input_index(self, kisc_code=None,
                                 start_year=None, last_year=None):

        mh_by_1dg = self.get_input_index('mh_input', 
                             kisc_code=kisc_code,
                             start_year=start_year, last_year=last_year)

        mm_by_1dg = self.get_input_index('mm_input',
                             kisc_code=kisc_code,
                             start_year=start_year, last_year=last_year)

        df = pd.merge(mh_by_1dg, mm_by_1dg, on=['year', 'mq', 'kisc_code'], sort=False)
        df['hh_input_index4'] =  100*df.mh_input_index4/df.mm_input_index4
        df['hh_input_rate4'] = df.mh_input_rate4 - df.mm_input_rate4
        return df[['year', 'mq', 'kisc_code', 'hh_input_index4', 'hh_input_rate4']]

        
    def get_labor_cost_index(self, type='labor_cost', *, kisc_code=None,
                             start_year=None, last_year=None):
        if type == 'labor_cost':
            data = pd.DataFrame(
                      self.session.query(Lps252.year, Lps252.mq, Lps252.kisc_code,
                                         Lps252.labor_cost_index2, Lps254.labor_cost_rate2)\
                               .filter(Lps252.year == Lps254.year)\
                               .filter(Lps252.mq == Lps254.mq)\
                               .filter(Lps252.kisc_code == Lps254.kisc_code)\
                               .filter(Lps252.year>=start_year)\
                               .filter(Lps252.year<=last_year)\
                               .all())

            return data.loc[data['kisc_code'].isin(kisc_code)]
            
        if type == 'hour_pay':
            data = pd.DataFrame(
                      self.session.query(Lps251.year, Lps251.mq, Lps251.kisc_code,
                                         Lps251.hour_pay_index, Lps253.hour_pay_rate)\
                               .filter(Lps251.year == Lps253.year)\
                               .filter(Lps251.mq == Lps253.mq)\
                               .filter(Lps251.kisc_code == Lps253.kisc_code)\
                               .filter(Lps251.year>=start_year)\
                               .filter(Lps251.year<=last_year)\
                               .all())
                               
            return data.loc[data['kisc_code'].isin(kisc_code)]

        if type == 'nom_wage':
            return self.calculate_nomwage_index(kisc_code=kisc_code,
                                    start_year=start_year,
                                    last_year=last_year)

    def calculate_nomwage_index(self, kisc_code=None,
                            start_year=None, last_year=None):

        hpay_index = self.get_labor_cost_index('hour_pay',
                             kisc_code=['TT_AOU', 'C', 'TS_OU'],
                             start_year=start_year, last_year=last_year)
        hh_index = self.get_input_index('hh_input', 
                             kisc_code=['TT_AOU', 'C', 'TS_OU'],
                             start_year=start_year, last_year=last_year)


        df = pd.merge(hpay_index, hh_index, on=['year', 'mq', 'kisc_code'], sort=False)
        df['nom_wage_index'] =  df.hh_input_index4*df.hour_pay_index/100
        df['nom_wage_rate'] = df.hh_input_rate4 + df.hour_pay_rate
        return df[['year', 'mq', 'kisc_code', 'nom_wage_index', 'nom_wage_rate']]

    def get_laborinput_level(self, var_name='mh_input_t', *, kisc_code=None,
                             start_year=None, last_year=None):
        if var_name == 'mh_input_t':
            data = pd.DataFrame(
                      self.session.query(Lps211.year, Lps211.mq, Lps211.kisc_code,
                                         Lps211.__dict__[var_name])\
                               .filter(Lps211.kisc_code == Lps211.kisc_code)\
                               .filter(Lps211.year>=start_year)\
                               .filter(Lps211.year<=last_year) \
                               .filter(Lps211.__dict__[var_name] != 0) \
                               .all())

            return data.loc[data['kisc_code'].isin(kisc_code)]