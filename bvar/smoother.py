import numpy as np
from numpy.random import randn
from numpy.linalg import cholesky, inv
from base import Smoother
from filter import KalmanFilter
from utils import Y_Dimension_Checker, cholx

class DisturbanceSmoother(Smoother):
    def smoothing(self, y, *, Z=None, alpha0=None, P0=None, T=None, R=None,
                  H=None, Q=None, a=None, K=None, F=None, L=None, v=None):
        '''
        (1) Observation Eq: y(t) = Z(t)*state(t) + e(t), e(t) ~ N(0,H(t))
        (2) Transition Eq: state(t) = T(t)*state(t) + R(t)n(t), n(t) ~ N(0,Q(t))
        :param y: ndarray, mxt, dependent variable in Observation equation (1)
        :param Z: ndarray, Z(t) in (1) Eq for t = 1..t_max
        :param alpha0: ndarray, initial value of state
        :param PO: ndarray, initial variance of state
        :param T: ndarray, T(t) in (2) Eq for t = 1..t_max
        :param R: ndarray, R(t) in (2) Eq for t = 1..t_max
        :param H: ndarray, nvariance of e(t) in (1) Eq for t = 1..t_max
        :param Q: ndarray, variance of n(t) in (2) Eq for t = 1..t_max
        :param a: ndarray, filtered state by Kalmanfilter
        :param K: ndarray, Kalmangain
        :param F: ndarray,
        :param L: ndarray,
        :param V: ndarray,
        '''
        self.m, self.t = y.shape
        _, self.k = alpha0.shape
        self.backward_recursion_to_estimate_w_hat(H, R, Z, v, F, L, K, Q)
        self.forward_recursion_to_estimate_alpha_hat(a, P0, T, R, Q)
        return self

    def backward_recursion_to_estimate_w_hat(self, H, R, Z, v, F, L, K, Q):
        t, k, m = self.t, self.k, self.m
        w_hat = np.zeros((t, m+k, 1))
        r = np.zeros((t+1, k, 1))
        for i in range(t-1, -1, -1):
            Zt = Z[i*m:(i+1)*m, :]
            Ht = H[i*m:(i+1)*m, :]
            Rt = R[i*k:(i+1)*k, :]
            Qt = Q[i*k:(i+1)*k, :]
            
            w_hat[i, : m, :] = np.dot(np.dot(Ht, inv(F[i])), v[i]) - \
                               np.dot(np.dot(Ht, K[i].T), r[i+1]) #e_hat:mx1
            w_hat[i, m:m+k, :] = np.dot(np.dot(Qt, Rt.T), r[i+1]) # n_hat: kx1
            r[i] = np.dot(np.dot(Zt.T, inv(F[i])), v[i]) + np.dot(L[i].T, r[i+1])

        self.w_hat = w_hat
        self.r = r
        return self

    def forward_recursion_to_estimate_alpha_hat(self, a, P0, T, R, Q):
        t, k, m = self.t, self.k, self.m
        alpha_hat = np.zeros((t + 1, k, 1))
        alpha_hat[0] = a[0] + np.dot(P0, self.r[0])

        for i in range(t):
            Rt = R[i * k:(i + 1) * k, :]
            Qt = Q[i * k:(i + 1) * k, :]
            Tt = T[i * k:(i + 1) * k, :]
            alpha_hat[i + 1] = np.dot(Tt, alpha_hat[i]) + \
                               np.dot(np.dot(Rt, Qt), np.dot(Rt.T, self.r[i]))

        self.alpha_hat = alpha_hat
        return self

class DurbinKoopmanSmoother(Smoother):
    '''
       State Space Recursion Equations
       (1) Observation Eq: y(t) = Z(t)*state(t) + e(t) e(t) ~ N(0,H(t))
       (2) Transition Eq: state(t) = T(t)*state(t) + R(t)n(t) n(t) ~ N(0,Q(t))
        - wplus: ndarray, drawed random vector w+(w=(e',n')')
                 from density p(w)~N(0,diag{H1,...,Hn,Q1,...,Qn})
         m, k, t mean the dimension of y, state and the number of timeseries observation
        :param y: mxt or txm ndarray
        :param Z: (m*t)xk ndarray, Z(t) in (1) Eq for t = 1..t_max
        :param H: (m*t)xm ndarray, variance of e(t) in (1) Eq for t = 1..t_max
        :param Q: (k*t)xk ndarray, variance of n(t) in (2) Eq for t = 1..t_max
        :param T: (k*t)xk ndarray, T(t) in (2) Eq for t = 1..t_max
        :param R: (k*t)xk ndarray, R(t) in (2) Eq for t = 1..t_max
    '''
    def __init__(self, state0=None, state0_var=None):
        '''
        :param state0: ndarray, initial array of state
        :param state0_var: ndarray, initial variance of state 
        '''
        self.state0 = state0
        self.state0_var = state0_var
        self._kalmanfilter = KalmanFilter(state0=state0,
                                          state0_var=state0_var)
        self._smoother = DisturbanceSmoother()

    def draw_wplus(self, H, Q, s):
        ''' w = (e,n)' ~ p(w)
            p(w)~N(0, diag{H1, ..., Hn, Q1, ..., Qn})
        '''
        m, k, t = self.m, self.k, self.t
        if s is None:
            s = k
        if s > k:
            raise ValueError('s should not be greater than k')
        wplus = np.zeros(((m+s)*t, 1))
        mean = 0
        for i in range(t):
            Ht = H[i*m:(i+1)*m, :]
            Qt = Q[i*k:(i+1)*k, :][:s,:s]
            wplus[i*(m+s):i*(m+s)+m, :] = mean + \
                                          np.dot(cholesky(Ht).T, randn(m, 1))
            wplus[i*(m+s)+m:i*(m+s)+(m+s), :] = mean + \
                                                np.dot(cholesky(Qt).T, randn(s, 1))
        return wplus

    def state_space_recursion(self, wplus, Z, T=None, R=None):
        m, k, t = self.m, self.k, self.t
        mk = m + k
        state = np.zeros((k, t + 1))  # assume state0 ~ N(0,P1)
        y_plus = np.zeros((m, t))
        for i in range(t):

            Tt = T[i*k:(i + 1)*k, :]  # kxk
            Rt = R[i*k:(i + 1)*k, :]  # kxk
            et = wplus[i*mk:i*mk+m, :]
            nt = wplus[i*mk+m:i*mk+mk, :]
            Zt = Z[i*m:(i+1)*m, :]
            y_plus[:, i:i+1] = (np.dot(Zt, state[:, i:i+1]) + et)  # mx1.T = 1xm
            state[:, i:i+1] = (np.dot(Tt, state[:, i:i+1]) + np.dot(Rt, nt))  # kx1.T = 1xk

        self.y_plus, self.state_plus = y_plus, state
        return self

    def simulation_smoothing(self, y, *, Z=None, H=None, Q=None, T=None, R=None, s=None):
        self._kalmanfilter.filtering(y, Z=Z, H=H, Q=Q, T=T, R=R)
        filtered_state = self._kalmanfilter.state
        K, F, L, v = self._kalmanfilter.K, self._kalmanfilter.F, \
                     self._kalmanfilter.L, self._kalmanfilter.v
        self._smoother.smoothing(y, Z=Z, alpha0=self.state0, P0=self.state0_var, T=T,
                                 R=R, H=H, Q=Q, a=filtered_state, K=K, F=F, L=L, v=v)
        return self._smoother.w_hat, self._smoother.alpha_hat

    @Y_Dimension_Checker
    def smoothing(self, y, *, Z=None, T=None, R=None, H=None, Q=None, s=None):
        '''
         m, k, t mean the dimension of y, state and the number of timeseries observation
        :param y: mxt or txm ndarray
        :param Z: (m*t)xk ndarray, Z(t) in (1) Eq for t = 1..t_max
        :param H: (m*t)xm ndarray, variance of e(t) in (1) Eq for t = 1..t_max
        :param Q: (k*t)xk ndarray, variance of n(t) in (2) Eq for t = 1..t_max
        :param T: (k*t)xk ndarray, T(t) in (2) Eq for t = 1..t_max
        :param R: (k*t)xk ndarray, R(t) in (2) Eq for t = 1..t_max
        :param s: int, selection number
        '''

        self.m, self.t = y.shape
        _, self.k = Z.shape

        if self.state0 is None:
            self.state0 = np.zeros((self.k, 1))
        if self.state0_var is None:
            self.state0_var = np.zeros((self.k, self.k))

        if T is None:
            T = np.tile(np.eye(self._k), (self._t, 1)) #(kxt)xk_
        if R is None:
            R = np.tile(np.eye(self._k), (self._t, 1)) #(kxt)xk_

        self.w_hat, self.state_hat = \
            self.simulation_smoothing(y, Z=Z, H=H, Q=Q, T=T, R=R, s=s)
        self.loglik = self._kalmanfilter.loglik

        self.state_space_recursion(self.draw_wplus(H, Q, s), Z, T, R)
        self.w_hat_plus, self.state_hat_plus = \
            self.simulation_smoothing(self.y_plus, Z=Z, H=H, Q=Q, T=T, R=R, s=s)

        self.drawed_state = self.state_hat + self.state_plus - self.state_hat_plus
        return self

class CarterKohn(object):

    def __init__(self, state0, state0_var):
        self._kalmanfilter = KalmanFilter(state0=state0, state0_var=state0_var)

    def smoothing(self, y, *, Z=None, H=None, Q=None, T=None, R=None,
                 MU=None, s=None):
        '''
            Observation Eq: Y(t) = Z*state(t) + A*z(t) + e(t), var(e(t)) = H
            Transition Eq:  state(t) = MU + T*state(t-1) + R(t)n(t), var(n(t)) = Q
            - state: the kalman filtered state matrix
            - states_var: variance of the kalman-filtered state matrix
            - mu: constant term in the transition equation
            - T: coefficients of state in the transition equation
            - Q: variance of v(t) in the transition equation
            - s: number of specific variables to extract
            result:
            - generates drawed_state: sampling state matrix from normal
        '''
        self._kalmanfilter.filtering(y, Z=Z, H=H, Q=Q, T=T, R=R)
        state = self._kalmanfilter.state #txkx1
        state_var = self._kalmanfilter.state_var #kxk
        t = y.shape[0]
        k = state.shape[1]
        if s is None:
            s = k
        if MU is None:
            MU = np.zeros((k, 1))

        drawed_state = np.zeros((t, k, 1))
        wa = randn(k, t)
        p00 = np.squeeze(state_var[t-1, :s, :s])
        # drawed_state[t-1, :s, :] = state[t-1, :s, :] + np.dot(wa[t-1:t, :s], cholx(p00))
        drawed_state[t-1, :s, :] = state[t-1, :s, :] + np.dot(cholx(p00).T, wa[:s, t-1:t])

        for i in range(t-2, -1, -1):
            Ft = T[((t-2)-i)*k:((t-2)-i+1)*k, :][:s, :]
            Qt = Q[:s, :s]
            mu = MU[:s, :]
            pt = np.squeeze(state_var[i, :, :]) # kxk
            temp = np.dot(np.dot(pt, Ft.T), inv(np.dot(np.dot(Ft, pt), Ft.T)+Qt))
            mean = state[i] + \
                   np.dot(temp, (drawed_state[i+1, :s]-mu-np.dot(Ft, state[i, :])))
            variance = pt - np.dot(temp, np.dot(Ft, pt))
            drawed_state[i, :s] = mean[:s, :] + np.dot(cholx(variance[:s, :s]).T, wa[:s, t-1:t])

        self.drawed_state = drawed_state[:, :s, 0]
        return self