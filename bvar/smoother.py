import numpy as np
from numpy.random import randn
from numpy.linalg import cholesky, inv
from bvar.base import Smoother
from bvar.filter import KalmanFilter
from bvar.utils import NoneValueChecker, DimensionYChecker

class DisturbanceSmoother(Smoother):
    def smoothing(self, y, *, Z=None, alpha0=None, P0=None, T=None, R=None,
                  H=None, Q=None, a=None, K=None, F=None, L=None, v=None):
        '''
        (1) Observation Eq: y(t) = Z(t)*state(t) + e(t), e(t) ~ N(0,H(t))
        (2) Transition Eq: state(t) = T(t)*state(t) + R(t)n(t), n(t) ~ N(0,Q(t))
        :param y: nparray, mxt, dependent variable in Observation equation (1)
        :param Z: nparray, Z(t) in (1) Eq for t = 1..t_max
        :param alpha0: nparray, initial value of state
        :param PO: nparray, initial variance of state
        :param T: nparray, T(t) in (2) Eq for t = 1..t_max
        :param R: nparray, R(t) in (2) Eq for t = 1..t_max
        :param H: nparray, nvariance of e(t) in (1) Eq for t = 1..t_max
        :param Q: nparray, variance of n(t) in (2) Eq for t = 1..t_max
        :param a: nparray, filtered state by Kalmanfilter
        :param K: nparray, Kalmangain
        :param F: nparray,
        :param L: nparray,
        :param V: nparray,
        '''
        self.m, self.t = y.shape
        self.k, _ = alpha0.shape
        self.backward_recursion_to_estimate_w_hat(H, R, Z, v, F, L, K, Q)
        self.forward_recursion_to_estimate_alpha_hat(a, P0, T, R, Q)
        return self

    def backward_recursion_to_estimate_w_hat(self, H, R, Z, v, F, L, K, Q):
        t, k, m = self.t, self.k, self.m
        w_hat = np.zeros((t, m+k, 1))
        r = np.zeros((t+1, k, 1))
        Rt = R
        for i in range(t-1, -1, -1):

            if m == 1: 
                Ht, Zt = H, Z
            else: 
                Ht = H[i*m:(i+1)*m, :]
                Zt = Z[i*m:(i+1)*m, :]

            w_hat[i, : m, :] = np.dot(np.dot(Ht, inv(F[i])), v[i]) - np.dot(np.dot(Ht, K[i].T), r[i+1]) #e_hat:mx1
            w_hat[i, m:m+k, :] = np.dot(np.dot(Q, Rt.T), r[i+1]) # n_hat: kx1
            r[i] = np.dot(np.dot(Zt.T, inv(F[i])), v[i]) + np.dot(L[i].T, r[i+1])

        self.w_hat = w_hat
        self.r = r
        return self

    def forward_recursion_to_estimate_alpha_hat(self, a, P0, T, R, Q):
        t, k, m = self.t, self.k, self.m
        alpha_hat = np.zeros((t + 1, k, 1))
        alpha_hat[0] = a[0] + np.dot(P0, r[0])
        
        Tt = T
        Rt = R
        for i in range(t):
            alpha_hat[i + 1] = np.dot(Tt, alpha_hat[i]) + \
                               np.dot(np.dot(Rt, Q), np.dot(Rt.T, self.r[i]))

        self.alpha_hat = alpha_hat
        return self

class DurbinKoopmanSmoother(Smoother):
    """
       State Space Recursion Equations
       (1) Observation Eq: y(t) = Z(t)*state(t) + e(t) e(t) ~ N(0,H(t))
       (2) Transition Eq: state(t) = T(t)*state(t) + R(t)n(t) n(t) ~ N(0,Q(t))
        - wplus: nparray, drawed random vector w+(w=(e',n')') from density p(w)~N(0,diag{H1,...,Hn,Q1,...,Qn})
        - m: int, the number of dimension of y
        - k: int, the number of dimension(elements) in alpha(=state)
        - t: int, the number of observation time
        - Z: nparray, Z(t) in (1) Eq for t = 1..t_max
        - T: nparray, T(t) in (2) Eq for t = 1..t_max
        - R: nparray, R(t) in (2) Eq for t = 1..t_max
    """
    def __init__(self, state0=None, state0_var=None):
        self.state0 = state0
        self.state0_var = state0_var
        self._kalmanfilter = KalmanFilter(state0=state0, state0_var=state0_var)
        self._smoother = DisturbanceSmoother()

    def draw_wplus(self, H, Q):
        ''' w = (e,n)' ~ p(w) 
            p(w)~N(0, diag{H1, ..., Hn, Q1, ..., Qn})
        '''
        m, k, t = self.m, self.k, self.t
        wplus = np.zeros(((m+k)*t, 1))
        mean = 0

        for i in range(t):

            if m == 1: Ht, Qt = H, Q
            else: Ht, Qt = H[i*m:(i+1)*m, :], Q[i*k:(i+1)*k, :]
            wplus[i*(m+k):i*(m+k)+m, :] = mean + \
                                          np.dot(cholesky(Ht).T,randn(m,1))
            wplus[i*(m+k)+m:i*(m+k)+(m+k), :] = mean + \
                                                np.dot(cholesky(Qt).T, randn(k,1))
        return wplus

    def state_space_recursion(self, wplus, Z, T=None, R=None):
        '''
        -wplus: nparray, drawed random vector w+(w=(e',n')') from density p(w)~N(0,diag{H1,...,Hn,Q1,...,Qn})
        -m: int, the number of dimension of y
        -k: int, the number of dimension(elements) in alpha(=state)
        -t: int, the number of observation time
        -Z: nparray, Z(t) in (1) Eq for t = 1..t_max
        -T: nparray, T(t) in (2) Eq for t = 1..t_max
        -R: nparray, R(t) in (2) Eq for t = 1..t_max
        '''
        m, k, t = self.m, self.k, self.t
        if T is None:
            Tt = np.eye(k)
        if R is None:
            Rt = np.eye(k)

        mk = m + k
        state = np.zeros((k, t + 1))  # assume state0 ~ N(0,P1)
        y_plus = np.zeros((m, t))

        for i in range(t):

            et = wplus[i * mk:i * mk + m, :]
            nt = wplus[i * mk + m:i * mk + mk, :]
            Zt = Z[i * m:(i + 1) * m, :]  # mxk
            y_plus[:, i] = (np.dot(Zt, state[:, i]) + et).T  # mx1.T = 1xm
            state[:, i + 1] = (np.dot(Tt, state[:, i]) + np.dot(Rt, nt)).T  # kx1.T = 1xk

        self.y_plus, self.state_plus = y_plus, state
        return self

    def simulation_smoothing(self, y, *, Z=None, H=None, Q=None, T=None, R=None):

        self._kalmanfilter.filtering(y, Z=Z, H=H, Q=Q, T=T, R=R)
        filtered_state = self._kalmanfilter.state
        K, F, L, v = self._kalmanfilter.K, self._kalmanfilter.F, \
                     self._kalmanfilter.L, self._kalmanfilter.v
        self._smoother.smoothing(y, Z=Z, alpha0=self.state0, P0=self.state0_var, T=T,
                                R=R, H=H, Q=Q, a=filtered_state, K=K, F=F, L=L, v=v)
        return self._smoother.w_hat, self._smoother.alpha_hat

    @DimensionYChecker
    def smoothing(self, y, *, Z=None, T=None, R=None, H=None, Q=None):

        self.m, self.t = y.shape
        
        _, self.k = self.Z

        if self.state0 is None:
            self.state0 = np.zeros((self.k, 1))
        if self.state0_var is None:
            self.state0_var = np.zeros((self.k,self.k))

        self.w_hat, self.state_hat = \
            self.simulation_smoothing(y, Z=Z, H=H, Q=Q, T=T, R=R)
        self.loglik = self._kalmanfilter.loglik

        self.state_space_recursion(self.draw_wplus(H, Q), Z, T, R)
        self.w_hat_plus, self.state_hat_plus = \
            self.simulation_smoothing(self.y_plus, Z=Z, H=H, Q=Q, T=T, R=R)

        self.state_tilda = self.state_hat + self.state_plus - self.state_hat_plus
        return self

class CarterKohn(object):
    
    def __init__(self, state0, state0_var):
        self._kalmanfilter = KalmanFilter(state0=state0, state0_var=state0_var)
        
    def estimate(self, *, Z=None, H=None, Q=None, T=None, R=None
                       MU=None, s=None):
        '''
            Observation Eq: Y(t) = Z*state(t) + A*z(t) + e(t), var(e(t)) = H
            Transition Eq:  state(t) = MU + T*state(t-1) + R(t)n(t), var(n(t)) = Q  
            - state: the kalman filtered state matrix
            - states_var: variance of the kalman filtered state matrix
            - mu: constant term in the transition equation
            - T: coefficients of state in the transition equation 
            - Q: variance of v(t) in the transition equation
            - s: number of specific variables to extract 
            result:
            - generates drawed_state: sampling state matrix from normal
        '''
        self._kalmanfilter.filtering(y, Z=Z, H=H, Q=Q, T=T, R=R)
        state = self._kalmanfilter.state
        state_var = self._kalmanfilter.state_var

        t, ns = state.shape
        if t < ns:
            state = state.T
            t, ns = state.shape
        if s is None: 
            s = ns
        if MU is None:
            MU = np.zeros((1,ns))
            
        drawed_state = np.zeros((t,ns))
        wa = randn(t,ns)
        f = T[:s,:] #sxns
        q = Q[:s,:s]
        mu = MU[:,:s]
        p00 = np.squeeze(state_var[t-1,:s,:s])
        drawed_state[t-1,:s] = state[t-1,:s] + np.dot(wa[t-1,:s],cholx(p00))

        for i in range(t-2,0,-1):
            
            pt = squeeze(state_var[i,:,:]) # nsxns
            temp = np.dot(np.dot(pt,f.T),inv(np.dot(np.dot(f,pt),f.T))+q)
            mean = state[i,:] + np.dot(temp,(drawed_state[i+1,:s]-
                                             mu-np.dot(state[i,:],f.T)).T).T
            variance = pt - np.dot(temp,np.dot(f,pt))
            drawed_state[i,:s] = mean[:,:s] + np.dot(wa[i,:s],cholx(variance[:s,:s]))
        
        self.drawed_state = drawed_state[:,:s]
        return self