import numpy as np
from numpy import ndarray, full, r_, tile, mean, \
                  std, real, nan, empty, vstack, \
                  atleast_2d, sqrt, dot, argsort, \
                  sort, absolute, eye, zeros
from numpy.linalg import cholesky, eig, eigvals
from scipy.linalg import sqrtm
from weakref import WeakKeyDictionary
from pandas import read_excel, ExcelWriter
from functools import wraps

def array_checker(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        arr = args[0]
        if not isinstance(arr, ndarray):
            arr = atleast_2d(arr)
        T, N = arr.shape
        if T < N:
            arr = arr.T
        return func(arr,**kwargs)
    return wrapper

class NoneValueChecker(object):
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        for name, value in kwargs.items():
            print("key:{0}, value:{1}".format(name,value))
            if value is None:
                raise ValueError('The keyword input "{0}" must \
                    be specified, not None value'.format(name))
        return self.func(*args, **kwargs)

class Y_Dimension_Checker(object):
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        pargs = list()
        for arg in args:
            if isinstance(arg, np.ndarray):
                m, t = arg.shape
                if m > t: pargs.append(arg.T)
                else: pargs.append(arg)
        return self.func(*pargs, **kwargs)

def is_coefficient_stable(coef, n, l):
    '''
        Input
         - coef: coefficients
         - n: number of endog variables(dimension of Y)
         - l: number of lags
        Output
         - if ee<1 stable(True), else, not stable(False), S=0
    '''
    FF = zeros((n * l, n * l))
    FF[n:n * l, :n * (l - 1)] = eye(n * (l - 1))
    FF[:n, :n * l] = coef.reshape((n * l + 1, n))[:n * l, :n].T  # coef.reshape((n*l+1,n)): 7*3
    ee = max(absolute(eigvals(FF)))
    return ee < 1

@array_checker
def get_principle_component(data, k):
    '''
    Input:
     - data: matrix data(TxN , N:number of variable,dimension) with column mean zero
     - k: k th principle component
    Output:
     -factor:k th principle component of the data
     -lamda: loadings
    '''
    T, N = data.shape
    xx = dot(data.T, data)
    eval, evec = eig(xx) # NxN
    index = argsort(eval, axis=0)[::-1] # sorting index of eigenvalues from the highest to the loweset eigenvalue
    sorted_eval = sort(eval, axis=0)[::-1] # sorting eigenvalues from the highest to the loweset one
    reord_evec = evec[:, index]
    lamda = sqrt(N)*reord_evec[:,:k] # NxK
    factor = dot(data,lamda)/N #Transfromed data TxN * NxK = TxK
    return factor, lamda

@array_checker
def standardize(arr):
    '''
    Input:
     - arr: data TxN matrix; T(number of observation), N(number of variable or dimension)
    Output: standardised matrix
    '''
    T, N = arr.shape
    col_mean = tile(mean(arr,axis=0),(T,1)) # axis=0 mean column
    col_std  = tile(std(arr,axis=0),(T,1))  # axis=0 mean column
    return (arr - col_mean)/col_std

@array_checker
def lag(arr,L=0):
    '''
    Input:
     - arr: Array matrix
     - L: number of lag that you want to shift the arr, default=0
    Output:
     - L_arr: shifted arr
    '''
    T, N = arr.shape
    if L >= 0:
        L_arr = r_[full((L, N),nan),arr[:T-L,:]]
    return L_arr

def cholx(x):
    try:
        return cholesky(x)
    except Exception:
        return real(sqrtm(x)).T

def vec(data):
    np,m = data.shape
    vec = empty((0,1))
    for i in range(m):
        data_col = atleast_2d(data[:,i]).T
        vec = vstack((vec,data_col))
    return vec

class DataChecker(object):
    def __init__(self):
        self._datas = WeakKeyDictionary()
    def __get__(self,instance,cls):
        if instance is None:
            return self
        return self._datas.get(instance)
    def __set__(self,instance,value):
        data = value
        if self.check_2dimension(data) is False:
            data = atleast_2d(data).T
        self.check_univariate_tseries(data)
        self._datas[instance] = data
    @staticmethod
    def check_univariate_tseries(value):
        if value.shape[1] > 1:
            raise ValueError('Time-Series data must be univariate(demension=1)')
    @staticmethod
    def check_2dimension(value):
        try:
            value.shape[1]
            return True
        except IndexError:
            return False


class DataImporterExporter(object):
    _file_path = None
    _file_name = None

    def __init__(self,path,name):
        DataImporterExporter._file_path = path
        if DataImporterExporter.is_file_extension_xlsx_or_xls(name):
            DataImporterExporter._file_name = name

    @classmethod
    def read_data(cls,sheet_name='Sheet1'):
        pass

    @classmethod
    def write_data(cls,sheet_name='Sheet1'):
        pass

    @classmethod
    def get_file_path_name(cls):
        from os import getcwd
        if getcwd() != DataImporterExporter._file_path:
            return ''.join([DataImporterExporter._file_path, '\\', DataImporterExporter._file_name])
        else:
            return DataImporterExporter._file_name
    @classmethod
    def is_file_extension_xlsx_or_xls(cls,name):
        if ('.xlsx' in name) or ('.xls' in name):
            return True
        else:
            raise ValueError('Excel filename extension(".xlsx"or".xls") must be included')

class ExcelImporter(DataImporterExporter):
    def __init__(self, path, name):
        super(ExcelImporter, self).__init__(path, name)

    @classmethod
    def read_data(cls, sheet_name='Sheet1'):
        data = read_excel(ExcelImporter.get_file_path_name(), sheetname=sheet_name)
        return data


class ExcelExporter(DataImporterExporter):
    _writer = None
    _dataframe = None

    def __init__(self, path, name, dataframe):
        super(ExcelExporter, self).__init__(path, name)
        ExcelExporter._dataframe = dataframe

    @classmethod
    def write_data(cls, sheetname='Sheet1'):
        ExcelExporter._writer = ExcelWriter(ExcelExporter.get_file_path_name())
        ExcelExporter._dataframe.to_excel(ExcelExporter._writer, sheet_name=sheetname)
        ExcelExporter._writer.save()

class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
