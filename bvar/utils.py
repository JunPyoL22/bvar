import numpy as np
from numpy.linalg import cholesky, eig, eigvals
from scipy.linalg import sqrtm
from weakref import WeakKeyDictionary
from pandas import read_excel, ExcelWriter
from functools import wraps

def array_checker(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        new_args = list(args)
        arr = new_args.pop(0)
        if not isinstance(arr, np.ndarray):
            arr = np.atleast_2d(arr)
        T, N = arr.shape
        if T < N:
            arr = arr.T
        new_args.insert(0, arr)
        return func(*new_args, **kwargs)
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

    def __get__(self, instance, owner):
        self.cls = owner
        self.obj = instance
        return self.__call__

    def __call__(self, *args, **kwargs):
        pargs = list(args)
        for i, arg in enumerate(args):
            if isinstance(arg, np.ndarray):
                m, t = arg.shape
                if m > t:
                    pargs[i] = arg.T
        if self.obj:
            return self.func.__call__(self.obj, *pargs, **kwargs)
        return self.func.__call__(*pargs, **kwargs)

def is_coefficient_stable(coef, n, l):
    '''
        Input
         - coef: coefficients
         - n: number of endog variables(dimension of Y)
         - l: number of lags 
        Output
         - if ee<1 stable(True), else, not stable(False), S=0
    '''
    if coef.shape[0]/n == n*l:
        reshaped_coef = coef.reshape((n*l, n))
    else:
        reshaped_coef = coef.reshape(n*l+1, n)
    FF = np.zeros((n * l, n * l))
    FF[n:n * l, :n * (l - 1)] = np.eye(n * (l - 1))
    # Assume last "n" elements of the coef vector represents coefficients of constant term
    FF[:n, :n * l] = reshaped_coef[:n * l, :n].T  # coef.reshape((n*l+1,n)): 7*3
    ee = max(np.absolute(eigvals(FF)))
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
    xx = np.dot(data.T, data)
    eval, evec = eig(xx) # NxN
    index = np.argsort(eval, axis=0)[::-1] # sorting index of eigenvalues from the highest to the loweset eigenvalue
    sorted_eval = np.sort(eval, axis=0)[::-1] # sorting eigenvalues from the highest to the loweset one
    reord_evec = evec[:, index]
    lamda = np.sqrt(N)*reord_evec[:,:k] # NxK
    factor = np.dot(data,lamda)/N #Transfromed data TxN * NxK = TxK
    return factor, lamda

@array_checker
def standardize(arr):
    '''
    Input:
     - arr: data TxN matrix; T(number of observation), N(number of variable or dimension)
    Output: standardised matrix
    '''
    T, N = arr.shape
    col_mean = np.tile(np.mean(arr, axis=0), (T, 1)) # axis=0 mean column
    col_std  = np.tile(np.std(arr, axis=0), (T, 1))  # axis=0 mean column
    return (arr - col_mean)/col_std

@array_checker
def lag(arr, L=0):
    '''
    Input:
     - arr: Array matrix
     - L: number of lag that you want to shift the arr, default=0
    Output:
     - L_arr: shifted arr
    '''
    T, N = arr.shape
    if L >= 0:
        L_arr = np.r_[np.full((L, N), np.nan), arr[:T-L, :]]
    return L_arr

def cholx(x):
    try:
        return cholesky(x)
    except Exception:
        return np.real(sqrtm(x)).T

def vec(data):
    t, m = data.shape
    vec = np.empty((0, 1))
    for i in range(m):
        data_col = np.atleast_2d(data[:, i]).T
        vec = np.vstack((vec, data_col))
    return vec

class DataChecker(object):
    def __init__(self):
        self._datas = WeakKeyDictionary()

    def __get__(self, instance, cls):
        if instance is None:
            return self
        return self._datas.get(instance)

    def __set__(self, instance, value):
        data = value
        if self.check_2dimension(data) is False:
            data = np.atleast_2d(data).T
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

    def __init__(self, path, name):
        self._file_path = path
        if self.is_file_extension_xlsx_or_xls(name):
            self._file_name = name

    def read_data(self, sheet_name='Sheet1'):
        pass

    def write_data(self, dataframe, sheet_name='Sheet1'):
        pass

    def get_file_path_name(self):
        from os import getcwd
        if getcwd() != self._file_path:
            return ''.join([self._file_path, '\\', self._file_name])
        else:
            return self._file_name

    def is_file_extension_xlsx_or_xls(self, name):
        if ('.xlsx' in name) or ('.xls' in name):
            return True
        else:
            raise ValueError('Excel filename extension(".xlsx"or".xls") must be included')

class ExcelImporter(DataImporterExporter):
    def __init__(self, path, name):
        super(ExcelImporter, self).__init__(path, name)

    def read_data(self, sheet_name='Sheet1'):
        data = read_excel(self.get_file_path_name(), sheetname=sheet_name)
        return data

class ExcelExporter(DataImporterExporter):

    def __init__(self, path, name):
        super(ExcelExporter, self).__init__(path, name)
        self._writer = ExcelWriter(self.get_file_path_name())

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._writer.save()

    def write_data(self, dataframe, sheetname='Sheet1'):
        dataframe.to_excel(self._writer, sheet_name=sheetname)
        return self

class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
