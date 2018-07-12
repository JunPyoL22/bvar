from abc import ABCMeta, abstractmethod
from scipy.linalg import lstsq
import numpy as np

class BaseLinearRegression(object):

    def fit(self, Y, X, method='ls'):
        # self.Y, self.X = Y, X
        if method is 'ls':
            self.coef, _, _, _ = lstsq(X, Y)
            error = Y - np.dot(X, self.coef)
            self.sse = np.dot(error.T, error)
            self.sigma = (1/(Y.shape[0]-X.shape[1]))*self.sse
        return self

class BayesianModel(BaseLinearRegression):
    __metaclass__ = ABCMeta
    @abstractmethod
    def set_prior(self):
        pass

    @abstractmethod
    def get_posterior_distribution(self):
        pass

    @abstractmethod
    def gibbs_sampling(self):
        pass

    @abstractmethod
    def estimate(self):
        pass

class BasePrior(metaclass=ABCMeta):

    @abstractmethod
    def get_posterior_distribution(self):
        pass

class Filter(metaclass=ABCMeta):

    @abstractmethod
    def filtering(self):
        pass

class Smoother(metaclass=ABCMeta):

    @abstractmethod
    def smoothing(self):
        pass

class SetDataForVAR(object):

    def __init__(self, lag=0, const=True, forecast_method=None, forecast_horizon=None):
        '''
        :param lag: int, the number of lag order, default = 0 
        :param const: Boolean, constant term in X
        :param forecast_method: string, 'Direct' or 'Iterate'
        :param forecast_horizon: int, the number of forecasting horizon
        '''
        self.lag = lag
        self.const = const

        if self.__is_forecasting(forecast_method, forecast_horizon):
            self.forecast_method = forecast_method
            self.forecast_horizon = forecast_horizon

    def prepare(self, Y):
        self.t = -self.lag
        if self.forecast_method is not None:
            data = self._setup_Y_for_forecasting(Y)
            self.Y = data.Y1
            self._setup_X_on_VAR(data.Y2, const=self.const)
            if self.forecast_method is 'Direct':
                self.t += data.t - 1
            else:
                self.t += data.t - self.forecast_horizon
        else:
            self.Y = Y[self.lag:, :]
            self._setup_X_on_VAR(Y, const=self.const)
            self.t += Y.shape[0]
        return self

    def _setup_Y_for_forecasting(self, Y):
        return SetupForForecasting(self.forecast_method,
                                   self.forecast_horizon).get_data(Y, self.lag)

    def _setup_X_on_VAR(self, Y, const):
        from bvar.utils import lag
        '''
        :param const: include a constant column in X or Not
        '''
        t = Y.shape[0]
        p = self.lag
        x = np.empty((t - p, 0))
        for i in range(1, p + 1):
            x_lag = lag(Y, L=i)[p:, :]
            x = np.append(x, x_lag, axis=1)
        if const:
            x = np.c_[np.ones((t - p, 1)), x]
        self.X = x
        return self

    def __is_forecasting(self, forecast_method, forecast_horizon):
        if forecast_method is not None:
            if forecast_method not in ['Direct', 'Iterate']:
                raise ValueError('forecasting method must be one of "Direct", "Iterate"')
        if forecast_horizon is not None:
            if forecast_horizon < 0:
                raise ValueError('forecasting horizon("h") shouldn\'t be less than 0')
        return True

class SetupForForecasting(object):

    def __init__(self, method_type, horizon):
        self.method_type = method_type
        self.h = horizon

    def get_data(self, Y, lag):
        '''
        Iterated forecasts:
            Y(t) = A0 + Y(t-1) x A1 + ... + Y(t-p) x Ap + e(t)
            so that in this case there are p lags of Y (from 1 to p).
        Direct h-step ahead foreacsts:
            Y(t+h) = A0 + Y(t) x A1 + ... + Y(t-p+1) x Ap + e(t+h)
            so that in this case there are also p lags of Y (from 0 to p-1).
        '''
        if self.method_type is 'Direct':
            return self._direct(Y, lag)
        return self._iterate(Y, lag)

    def _direct(self, Y, lag):
        self.t = Y.shape[0] - self.h - 1
        Y1 = Y[self.h:, :][lag:self.t, :]
        self.Y1 = Y1[:-self.h,:]
        self.Y2 = Y[1:-self.h, :][:-self.h,:]
        return self
        # return DotDict( { 'Y1': Y1[:-self._h,:], 'Y2': Y[1:-self._h, :][:-self._h,:], 't': t } )

    def _iterate(self, Y, lag):
        self.t = Y.shape[0]
        self.Y1 = Y[lag:self.t, :][:-self.h, :]
        self.Y2 =  Y[:-self.h, :]
        return self
        # return DotDict( { 'Y1': Y[nlag:t, :][:-self._h, :], 'Y2': Y[:-self._h, :], 't': t } )
