import numpy as np
from numpy.linalg import inv, det
from scipy.sparse import spdiags
from base import Filter
from utils import Y_Dimension_Checker

class HpFilter(Filter):

    def __init__(self, penalty_param=None, period_type=None):

        if period_type is None or period_type not in ["Year", "Quarter", "Month"]:
            raise ValueError(
                'period_type must be assigned with one of these options such that "Year","Quarter","Month"')
        self.period_type = period_type
        self.penalty_param = penalty_param

    def set_penalty_parameter_on_hp_filter(self):
        '''
        return:
        - penalty_param: penalty parameter; quaterly: 1,600, monthly: 100,000< <140,000, annualy: 6< <14
                    Based on Raven,Uhlig(2002): annualy=6.25, monthly = 129600
        '''
        parameters_by_period = {
                                'Year': 6.25,
                                'Quarter': 1600,
                                'Month': 129600
                                }
        self.penalty_param = parameters_by_period.get(self.period_type)

    def filtering(self, data):
        '''
            Hodric Prescott Filter
            Input:
            - data: nparray univariate time series Tx1
            Attributes:
            - trend: nparray, trend component in the univariate time series array, Tx1
            - cycle: nparray, cycle component in the univariate time series array, Tx1
        '''
        T, N = data.shape

        if self.penalty_param is None:
            self.set_penalty_parameter_on_hp_filter()

        a = np.array([self.penalty_param, -4 * self.penalty_param, ((6 * self.penalty_param + 1) / 2.)])
        d = np.tile(a, (T, 1))

        d[0, 1] = -2. * self.penalty_param
        d[T - 2, 1] = -2. * self.penalty_param
        d[0, 2] = (1 + self.penalty_param) / 2.  # diagonal
        d[T - 1, 2] = (1 + self.penalty_param) / 2.  # diagonal
        d[1, 2] = (5 * self.penalty_param + 1) / 2.  # diagonal
        d[T - 2, 2] = (5 * self.penalty_param + 1) / 2.  # diagonal

        F = spdiags(d.T, [-2, -1, 0], T, T)  # [-2,-1,0] means from the 2th lower diagonal to the main diagonal
        F = F + F.T

        self.trend = np.array(np.dot(inv(F.todense()), data))
        self.cycle = np.array(data - self.trend)
        return self

class KalmanFilter(Filter):
    '''
        State Space Recursion Equations
        (1) Observation Eq: y(t) = Z(t)*state(t) + e(t), e(t) ~ N(0,H(t))
        (2) Transition Eq: state(t) = T(t)*state(t) + R(t)n(t), n(t) ~ N(0,Q(t))
        (1) Observation Eq: y(t) = Z(t)*state(t) + e(t), e(t) ~ N(0,H(t))
        (2) Transition Eq: state(t) = T(t)*state(t) + R(t)n(t), n(t) ~ N(0,Q(t))
         m, k, t mean the dimension of y, state and the number of timeseries observation
        :param y: mxt or txm ndarray
        :param Z: (m*t)xk ndarray, Z(t) in (1) Eq for t = 1..t_max
        :param H: (m*t)xm ndarray, variance of e(t) in (1) Eq for t = 1..t_max
        :param Q: (k*t)xk ndarray, variance of n(t) in (2) Eq for t = 1..t_max
        :param T: (k*t)xk ndarray, T(t) in (2) Eq for t = 1..t_max
        :param R: (k*t)xk ndarray, R(t) in (2) Eq for t = 1..t_max
        Attributes
         - state: filteted state by Kalmanfilter
         - state_var: filteted variance of state by Kalmanfilter
         - loglik: loglikely value
         - Kt: Kalman Gain
         - Lt: 
         - Ft: variance of vt
         - vt: innovation
    '''
    def __init__(self, *, state0=None, state0_var=None):
        '''
        state0: nparray, initial mean, value or vector(1xk) of state when t=1
        state0_var: nparray, initial variance of state, value or matrix(kxk) of state when t=1
        '''
        self.state0 = state0
        self.state0_var = state0_var

    @Y_Dimension_Checker
    def filtering(self, y, *, Z=None, H=None, Q=None, T=None, R=None):
        '''
        (1) Observation Eq: y(t) = Z(t)*state(t) + e(t), e(t) ~ N(0,H(t))
        (2) Transition Eq: state(t) = T(t)*state(t) + R(t)n(t), n(t) ~ N(0,Q(t))
         m, k, t mean the dimension of y, state and the number of timeseries observation
        :param y: mxt or txm ndarray
        :param Z: (*t)xk ndarray, Z(t) in (1) Eq for t = 1..t_max
        :param H: (*t)xm ndarray, variance of e(t) in (1) Eq for t = 1..t_max
        :param Q: (k*t)xk ndarray, variance of n(t) in (2) Eq for t = 1..t_max
        :param T: (k*t)xk ndarray, T(t) in (2) Eq for t = 1..t_max
        :param R: (k*t)xk ndarray, R(t) in (2) Eq for t = 1..t_max
        '''
        self._m, self._t = y.shape
        self._k, _ = self.state0.shape
        if self._k == 1:
            self.state0 = self.state0.T #kx1
            self._k, _ = self.state0.shape

        if T is None:
            T = np.tile(np.eye(self._k), (self._t, 1)) #(kxt)xk_
        if R is None:
            R = np.tile(np.eye(self._k), (self._t, 1)) #(kxt)xk_

        self.forward_recursion_to_estimate_state(y, Z, T, R, H, Q)
        return self

    def _get_container(self):
        m, k, t,  = self._m, self._k, self._t

        if self.state0 is None:
            state = np.zeros((t + 1, k, 1))
        else:
            state = np.zeros((t + 1, k, 1))
            state[0] = self.state0

        state_var = np.zeros((t + 1, k, k))
        if self.state0_var is not None:
            state_var[0, :k, :k] = self.state0_var

        K = np.zeros((t, k, m))
        F = np.zeros((t, m ,m))
        L = np.zeros((t, k, k))
        v = np.zeros((t, m, 1))

        return state, state_var, K, F, L, v

    def forward_recursion_to_estimate_state(self, y, Z, T, R, H, Q):
        m, k, t,  = self._m, self._k, self._t
        loglik = 0
        alpha_t, Pt, Kt, Ft, Lt, vt = self._get_container()

        for i in range(t):
            yt = y[:, i:i+1]  # mx1
            Ht = H[i*m:(i + 1)*m, :]  #  mxm
            Zt = Z[i*m:(i + 1)*m, :]  #  mxk
            Tt = T[i*k:(i + 1)*k, :]  # kxk
            Rt = R[i*k:(i + 1)*k, :]  # kxk
            Qt = Q[i*k:(i + 1)*k, :]

            vt[i] = yt - np.dot(Zt, alpha_t[i])  # mx1
            Ft[i, :, :] = np.dot(np.dot(Zt, Pt[i]), Zt.T) + Ht  # mxm
            Kt[i, :, :] = np.dot(np.dot(Tt, Pt[i]), np.dot(Zt.T, inv(Ft[i])))  # kxm
            Lt[i, :, :] = Tt - np.dot(Kt[i], Zt)  # kxk
            alpha_t[i + 1] = np.dot(Tt, alpha_t[i]) + np.dot(Kt[i], vt[i])  # kx1
            Pt[i+1, :, :] = np.dot(np.dot(Tt, Pt[i]), Lt[i].T) + np.dot(np.dot(Rt, Qt), Rt.T)  # kxk
            loglik = loglik + np.log10(det(Ft[i])) + np.dot(np.dot(vt[i].T,
                                                                   inv(Ft[i])), vt[i])
        self.K = Kt
        self.F = Ft
        self.L = Lt
        self.v = vt
        self.loglik = -0.5 * loglik
        self.state = alpha_t
        self.state_var = Pt
        return self
