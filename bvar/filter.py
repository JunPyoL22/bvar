import numpy as np
from numpy.linalg import inv, det
from scipy.sparse import spdiags
from bvar.base import Filter

class HpFilter(Filter):
    # data = DataChecker()

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
        :param y: mxt, dependent variable in Observation equation (1)
        :param Z: Z(t) in (1) Eq for t = 1..t_max
        :param H: variance of e(t) in (1) Eq for t = 1..t_max
        :param Q: variance of n(t) in (2) Eq for t = 1..t_max
        :param T: T(t) in (2) Eq for t = 1..t_max
        :param R: R(t) in (2) Eq for t = 1..t_max
        Attributes
         - state: estimated state using Kalmanfilter
         - loglik: loglikely value
         - K:
         - L:
         - F:
         - v:
    '''

    def __init__(self, state0, state0_var):
        '''
        state0: nparray, initial mean, value or vector(1xk) of state when t=1
        state0_var: nparray, initial variance of state, value or matrix(kxk) of state when t=1
        '''
        self.state0 = state0
        self.state0_var = state0_var


    def filtering(self, y, Z, H, Q, T=None, R=None):
        '''
        (1) Observation Eq: y(t) = Z(t)*state(t) + e(t), e(t) ~ N(0,H(t))
        (2) Transition Eq: state(t) = T(t)*state(t) + R(t)n(t), n(t) ~ N(0,Q(t))
        :param y: mxt, dependent variable in Observation equation (1)
        :param Z: Z(t) in (1) Eq for t = 1..t_max
        :param H: variance of e(t) in (1) Eq for t = 1..t_max
        :param Q: variance of n(t) in (2) Eq for t = 1..t_max
        :param T: T(t) in (2) Eq for t = 1..t_max
        :param R: R(t) in (2) Eq for t = 1..t_max
        :return: v, F, K, L
        '''
        T, R, state = self.get_initial_parameters_in_transition_equation(y, T, R)
        self.forward_recursion_to_estimate_state(y, Z, state, T, R, H, Q)
        return self

    def get_initial_parameters_in_transition_equation(self, y, T, R):
        m, t = y.shape
        k, _ = self.state0.shape

        if T is None:
            T = np.eye(k * t)

        if R is None:
            R = np.eye(k * t)

        if self.state0 is None:
            a = np.zeros((t + 1, k, 1))
        else:
            if t > k != 1:
                self.state0 = self.state0.T
            a = np.atleast_3d(np.r_[self.state0, np.zeros((t, k))])  # t+1xmx1

        if self.state0_var is None:
            self.state0_var = np.zeros((k, k))
        return T, R, a

    def forward_recursion_to_estimate_state(self, y, Z, state, T, R, H, Q):
        m, k, t = self.m, self.k, self.t
        a = state
        Pt = self.state0_var  # kxk
        loglik = 0

        K = np.zeros((t,k,m))
        F = np.zeros((t,m,m))
        L = np.zeros((t,k,k))
        v = np.zeros((t,k,1))

        for i in range(t):

            yt = np.atleast_2d(y[:, i]).T  # mx1
            Ht = H[i * m:(i + 1) * m, :]  # mxm
            Zt = Z[i * m:(i + 1) * m, :]  # mxk
            Tt = T[i * k:(i + 1) * k, i * k:(i + 1) * k]  # kxk
            Rt = R[i * k:(i + 1) * k, i * k:(i + 1) * k]  # kxk
            v[i] = yt - np.dot(Zt, a[i])  # mx1
            F[i, :, :] = np.dot(np.dot(Zt, Pt), Zt.T) + Ht  # mxm
            K[i, :, :] = np.dot(np.dot(Tt, Pt), np.dot(Zt.T, inv(F[i])))  # kxm
            L[i, :, :] = Tt - np.dot(K[i], Zt)  # kxk
            a[i + 1] = np.dot(Tt, a[i]) + np.dot(K[i], v[i])  # kx1
            Pt = np.dot(np.dot(Tt, Pt), L[i].T) + np.dot(np.dot(Rt, Q), Rt.T)  # kxk
            loglik = loglik + np.log10(det(F[i])) + np.dot(np.dot(v[i].T, inv(F[i])), v[i])

        self.K = K
        self.F = F
        self.L = L
        self.v = v
        self.loglik = -0.5 * loglik
        self.state = a
        return self
