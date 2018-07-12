from sqlalchemy import Column
from sqlalchemy.dialects.mysql import CHAR, VARCHAR, DOUBLE
from sqlalchemy.ext.declarative import declarative_base

base = declarative_base()

class Lps003(base):
    '''
    Industrial production index
    '''
    __tablename__ = 'lps003'
    year = Column(CHAR,primary_key=True)
    mq = Column(VARCHAR,primary_key=True)
    kisc_code = Column(VARCHAR,primary_key=True)
    kisc_gbn = Column(CHAR)
    ip_index = Column(DOUBLE)

    def __new__(cls):
        print('{0} table created'.format(cls.__name__))
        return super('Lps003', cls).__new__(cls)

    def __init__(self,year,mq,kisc_code,kisc_gbn,ip_index):
        print('{0} table intialized'.format(self.__name__))
        self.year = year
        self.mq = mq
        self.kisc_code = kisc_code
        self.kisc_gbn = kisc_gbn
        self.ip_index = ip_index

    def __repr__(self): 
        return "<lps003('%s', '%s', '%s', '%s', '%f'>" %(self.year, self.mq, self.kisc_code, self.kisc_gbn, self.ip_index)
        
class Lps103(base):
    '''
    Industrial production rate
    '''
    __tablename__ = 'lps103'
    year = Column(CHAR,primary_key=True)
    mq = Column(VARCHAR,primary_key=True)
    kisc_code = Column(VARCHAR,primary_key=True)
    ip_index_rate = Column(DOUBLE)

    def __init__(self, year, mq, kisc_code, ip_index_rate):
        self.year = year
        self.mq = mq
        self.kisc_code = kisc_code
        self.ip_index_rate = ip_index_rate

class Lps141(base):
    '''
    Value Add Index
    '''
    __tablename__ = 'lps141'
    year = Column(CHAR,primary_key=True)
    mq = Column(VARCHAR,primary_key=True)
    kisc_code = Column(VARCHAR,primary_key=True)
    ca_gva_index = Column(DOUBLE)

    def __init__(self, year, mq, kisc_code, ca_gva_index):
        self.year = year
        self.mq = mq
        self.kisc_code = kisc_code
        self.ca_gva_index = ca_gva_index

class Lps142(base):
    '''
    Value Add Index (constant)
    '''
    __tablename__ = 'lps142'
    year = Column(CHAR,primary_key=True)
    mq = Column(VARCHAR,primary_key=True)
    kisc_code = Column(VARCHAR,primary_key=True)
    cp_gva_index = Column(DOUBLE)

    def __init__(self, year, mq, kisc_code, cp_gva_index):
        self.year = year
        self.mq = mq
        self.kisc_code = kisc_code
        self.cp_gva_index = cp_gva_index

class Lps143(base):
    '''
    Value Add rate
    '''
    __tablename__ = 'lps143'
    year = Column(CHAR,primary_key=True)
    mq = Column(VARCHAR,primary_key=True)
    kisc_code = Column(VARCHAR,primary_key=True)
    ca_gva_rate = Column(DOUBLE)

    def __init__(self, year, mq, kisc_code, ca_gva_rate):
        self.year = year
        self.mq = mq
        self.kisc_code = kisc_code
        self.ca_gva_rate = ca_gva_rate

class Lps144(base):
    '''
    Value Add rate (constant)
    '''
    __tablename__ = 'lps144'
    year = Column(CHAR,primary_key=True)
    mq = Column(VARCHAR,primary_key=True)
    kisc_code = Column(VARCHAR,primary_key=True)
    cp_gva_rate = Column(DOUBLE)

    def __init__(self, year, mq, kisc_code, cp_gva_rate):
        self.year = year
        self.mq = mq
        self.kisc_code = kisc_code
        self.cp_gva_rate = cp_gva_rate

class Lps211(base):
    __tablename__ = 'lps211'
    year = Column(CHAR,primary_key=True)
    mq = Column(VARCHAR,primary_key=True)
    kisc_code = Column(VARCHAR,primary_key=True)
    mh_input = Column(DOUBLE)
    mh_input_t = Column(DOUBLE)

    def __init__(self,year,mq,kisc_code, mh_input, mh_input_t):
        self.year = year
        self.mq = mq
        self.kisc_code = kisc_code
        self.mh_input = mh_input
        self.mh_input_t = mh_input_t
        
    def __repr__(self): 
        return "<lps211('%s', '%s', '%s', '%f', '%f'>" \
        %(self.year, self.mq, self.kisc_code, self.mh_input, self.mh_input_t)


class Lps212(base):
    __tablename__ = 'lps212'
    year = Column(CHAR,primary_key=True)
    mq = Column(VARCHAR,primary_key=True)
    kisc_code = Column(VARCHAR,primary_key=True)
    mm_input = Column(DOUBLE)
    mm_input_t = Column(DOUBLE)

    def __init__(self, year, mq, kisc_code, mm_input, mm_input_t):
        self.year = year
        self.mq = mq
        self.kisc_code = kisc_code
        self.mm_input = mm_input
        self.mm_input_t = mm_input_t


class Lps221(base):
    __tablename__ = 'lps221'
    year = Column(CHAR, primary_key=True)
    mq = Column(VARCHAR, primary_key=True)
    kisc_code = Column(VARCHAR, primary_key=True)
    mh_input_index4 = Column(DOUBLE)

    def __init__(self, year, mq, kisc_code, mh_input_index4):
        self.year = year
        self.mq = mq
        self.kisc_code = kisc_code
        self.mh_input_index4 = mh_input_index4


class Lps222(base):
    __tablename__ = 'lps222'
    year = Column(CHAR, primary_key=True)
    mq = Column(VARCHAR, primary_key=True)
    kisc_code = Column(VARCHAR, primary_key=True)
    mm_input_index4 = Column(DOUBLE)

    def __init__(self, year, mq, kisc_code, mm_input_index4):
        self.year = year
        self.mq = mq
        self.kisc_code = kisc_code
        self.mm_input_index4 = mm_input_index4


class Lps223(base):
    __tablename__ = 'lps223'
    year = Column(CHAR, primary_key=True)
    mq = Column(VARCHAR, primary_key=True)
    kisc_code = Column(VARCHAR, primary_key=True)
    mh_input_rate4 = Column(DOUBLE)

    def __init__(self, year, mq, kisc_code, mh_input_rate4):
        self.year = year
        self.mq = mq
        self.kisc_code = kisc_code
        self.mh_input_rate4 = mh_input_rate4
        
    def __repr__(self): 
        return "<lps211('%s', '%s', '%s', '%f'>" %(self.year, self.mq, self.kisc_code, self.mh_input_t)

class Lps224(base):
    __tablename__ = 'lps224'
    year = Column(CHAR,primary_key=True)
    mq = Column(VARCHAR,primary_key=True)
    kisc_code = Column(VARCHAR,primary_key=True)
    mm_input_rate4 = Column(DOUBLE)

    def __init__(self, year, mq, kisc_code, mm_input_rate4):
        self.year = year
        self.mq = mq
        self.kisc_code = kisc_code
        self.mm_input_rate4 = mm_input_rate4


class Lps231(base):
    '''
    Industrial production prductivity index
    '''
    __tablename__ = 'lps231'
    year = Column(CHAR,primary_key=True)
    mq = Column(VARCHAR,primary_key=True)
    kisc_code = Column(VARCHAR,primary_key=True)
    mh_prod_index4 = Column(DOUBLE)

    def __init__(self, year, mq, kisc_code, mh_prod_index4):
        self.year = year
        self.mq = mq
        self.kisc_code = kisc_code
        self.mh_prod_index4 = mh_prod_index4


class Lps233(base):
    '''
    Industrial production prductivity rate
    '''
    __tablename__ = 'lps233'
    year = Column(CHAR,primary_key=True)
    mq = Column(VARCHAR,primary_key=True)
    kisc_code = Column(VARCHAR,primary_key=True)
    mh_prod_rate4 = Column(DOUBLE)

    def __init__(self, year, mq, kisc_code, mh_prod_rate4):
        self.year = year
        self.mq = mq
        self.kisc_code = kisc_code
        self.mh_prod_rate4 = mh_prod_rate4


class Lps241(base):
    '''
    Value Added prductivity Index
    '''
    __tablename__ = 'lps241'
    year = Column(CHAR,primary_key=True)
    mq = Column(VARCHAR,primary_key=True)
    kisc_code = Column(VARCHAR,primary_key=True)
    mh_add_index = Column(DOUBLE)

    def __init__(self, year, mq, kisc_code, mh_add_index):
        self.year = year
        self.mq = mq
        self.kisc_code = kisc_code
        self.mh_add_index = mh_add_index


class Lps243(base):
    '''
    Value Added prductivity rate
    '''
    __tablename__ = 'lps243'
    year = Column(CHAR,primary_key=True)
    mq = Column(VARCHAR,primary_key=True)
    kisc_code = Column(VARCHAR,primary_key=True)
    mh_add_rate = Column(DOUBLE)

    def __init__(self, year, mq, kisc_code, mh_add_rate):
        self.year = year
        self.mq = mq
        self.kisc_code = kisc_code
        self.mh_add_rate = mh_add_rate


class Lps251(base):
    '''
    labor cost Index
    '''
    __tablename__ = 'lps251'
    year = Column(CHAR, primary_key=True)
    mq = Column(VARCHAR, primary_key=True)
    kisc_code = Column(VARCHAR, primary_key=True)
    hour_pay_index = Column(DOUBLE)

    def __init__(self, year, mq, kisc_code, hour_pay_index):
        self.year = year
        self.mq = mq
        self.kisc_code = kisc_code
        self.hour_pay_index = hour_pay_index


class Lps252(base):
    '''
    labor cost Index
    '''
    __tablename__ = 'lps252'
    year = Column(CHAR, primary_key=True)
    mq = Column(VARCHAR, primary_key=True)
    kisc_code = Column(VARCHAR, primary_key=True)
    labor_cost_index2 = Column(DOUBLE)

    def __init__(self, year, mq, kisc_code, labor_cost_index2):
        self.year = year
        self.mq = mq
        self.kisc_code = kisc_code
        self.labor_cost_index2 = labor_cost_index2

class Lps253(base):
    '''
    labor cost Index
    '''
    __tablename__ = 'lps253'
    year = Column(CHAR, primary_key=True)
    mq = Column(VARCHAR, primary_key=True)
    kisc_code = Column(VARCHAR, primary_key=True)
    hour_pay_rate = Column(DOUBLE)

    def __init__(self, year, mq, kisc_code, hour_pay_rate):
        self.year = year
        self.mq = mq
        self.kisc_code = kisc_code
        self.hour_pay_rate = hour_pay_rate


class Lps254(base):
    '''
    labor cost index rate
    '''
    __tablename__ = 'lps254'
    year = Column(CHAR, primary_key=True)
    mq = Column(VARCHAR, primary_key=True)
    kisc_code = Column(VARCHAR, primary_key=True)
    labor_cost_rate2 = Column(DOUBLE)

    def __init__(self, year, mq, kisc_code, labor_cost_rate2):
        self.year = year
        self.mq = mq
        self.kisc_code = kisc_code
        self.labor_cost_rate2 = labor_cost_rate2