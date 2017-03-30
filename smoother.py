from filter import KalmanFilter
from numpy.random import randn
from numpy.linalg import cholesky
from numpy import dot, atleast_2d, zeros, eye, inv

class Smoother(object):
    def apply_smoother(self):
        raise NotImplementedError

class DurbinKoopmanSmoother(Smoother):
    """
    State Space Recursion Equations
    (1) Observation Eq: y(t) = Z(t)*alpha(t) + e(t) e(t) ~ N(0,H(t))
    (2) Transition Eq: alpha(t) = T(t)*alpha(t) + R(t)n(t) n(t) ~ N(0,Q(t))
    Input:
     - wplus: drawed random vector w+(w=(e',n')') from density p(w)~N(0,diag{H1,...,Hn,Q1,...,Qn})
     -p: the number of dimension of y
     -m: the number of dimension(elements) in alpha(=state)
     -t: Time
     -Z: Z(t) in (1) Eq for t = 1..t_max
     -T: T(t) in (2) Eq for t = 1..t_max
     -R: R(t) in (2) Eq for t = 1..t_max
    """
    def __init__(self):
        self._a = None
        self._alpha_hat = None
        self._alpha_plus = None
        self._alpha_hat_plus = None
        self._a_tilda = None
        self._w_hat = None
        self._w_hat_plus = None
        self._loglik = None
        self._y = None
        self._y_plus = None
        self._km_filter = KalmanFilterDK()

    @property
    def data(self):
        return self._y

    @property
    def a_tilda(self):
        return self._alpha_hat + self._alpha_plus - self._alpha_hat_plus

    @property
    def loglikely(  self):
        return self._loglik

    @data.setter
    def data(self, value):
        if self.check_2dimension(value) is False:
            data = atleast_2d(value).T
        p, t = data.shape
        if t < p:
            data = data.T
        self._y = data

    def apply_smoother(self, p, m, t, Z, H, Q, T=None, R=None, alpha0=None, P0=None):
        # p: dimension of Y(observation)
        # m: dimension of state
        w_plus = self._draw_w_plus(p, m, t, H, Q)
        self.state_space_recursion(w_plus, p, m, t, Z, T, R)
        self._w_hat, self._alpha_hat = self._km_filter.apply_filter(self._y, Z, H, Q, T, R, alpha0, P0)
        self._loglik = self._km_filter.loglikely
        self._w_hat_plus, self._alpha_hat_plus = self._km_filter.apply_filter(self._y_plus, Z, H, Q, T, R, alpha0, P0)

    def state_space_recursion(self, wplus, p, m, t, Z, T=None, R=None):
        if T is None:
            T = eye(m * t)
        if R is None:
            R = eye(m * t)
        pm = p + m
        alpha = zeros((m, t + 1))  # assume alpha1 ~ N(0,P1)
        y_plus = zeros((p, t))
        for i in range(t):
            et = wplus[i * pm:i * pm + p, :]
            nt = wplus[i * pm + p:i * pm + pm, :]
            Tt = T[i * m:(i + 1) * m, i * m:(i + 1) * m]
            Rt = R[i * m:(i + 1) * m, i * m:(i + 1) * m]
            Zt = Z[i * p:(i + 1) * p, :]  # pxm
            y_plus[:, i] = (dot(Zt, alpha[:, i]) + et).T  # px1.T = 1xp
            alpha[:, i + 1] = (dot(Tt, alpha[:, i]) + dot(Rt, nt)).T  # mx1.T = 1xm
        self._y_plus, self._alpha_plus = y_plus, alpha

    @staticmethod
    def draw_w_plus(p, m, t, H, Q):
        w_plus = zeros(((p+m)*t,1))
        for i in range(t):
            Hchol = cholesky(H[i*p:(i+1)*p,:])
            Qchol = cholesky(Q[i*m:(i+1)*m,:])
            #p(w)~N(0, diag{H1, ..., Hn, Q1, ..., Qn})
            w_plus[i*(p+m):i*(p+m)+p,:] = 0 + dot(Hchol.T,randn(p,1))
            w_plus[i*(p+m)+p:i*(p+m)+(p+m),:] = 0 + dot(Qchol.T, randn(m,1))
        return w_plus

    @staticmethod
    def check_2dimension(value):
        try:
            value.shape[1]
            return True
        except IndexError:
            return False


class KalmanFilterDK(KalmanFilter):
    def __init__(self):
        super(KalmanFilterDK, self).__init__()
        self._smoother = DisturbanceSmootherDK()

    def apply_filter(self, y, Z, H, Q, T=None, R=None, alpha0=None, P0=None):
        a  = self._filtered_state
        T, R, P0, alpha = self.__set_initial_parms_in_transition_equation(alpha0, P0, T, R)
        v, F, K, L = self.__forward_recursion_to_evaluate_state(y, alpha, P0, H, Z, T, R, Q)
        self._smoother.apply_smoother(y, alpha0, a, v, F, K, L, P0, Z, H, T, R, Q)
        w_hat, alpha_hat = self._smoother.w_hat, self._smoother.alpha_hat
        return w_hat, alpha_hat


class DisturbanceSmootherDK(Smoother):
    def __init__(self):
        self._r = None
        self._w_hat = None
        self._alpha_hat = None
        self._t = None
        self._m = None
        self._p = None

    @property
    def w_hat(self):
        return self._w_hat

    @property
    def alpha_hat(self):
        return self._alpha_hat

    def apply_smoother(self, y, alpha0, a, v, F, K, L, P0, Z, H, T, R, Q):
        self._p, self._t = y.shape
        self._m, _ = alpha0.shape
        self.__backward_recursion_to_evaluate_w_hat(H, R, Z, v, F, L, K, Q)
        self.__forward_recursion_to_evaluate_alpha_hat(a, P0, T, R, Q)

    def __backward_recursion_to_evaluate_w_hat(self, H, R, Z, v, F, L, K, Q):
        t, m, p = self._t, self._m, self._p
        w_hat = zeros((t, p+m, 1))
        r = zeros((t+1, m, 1))
        for i in range(t-1, -1, -1):
            Ht = H[i*p:(i+1)*p, :]
            Rt = R[i*m:(i+1)*m, i*m:(i+1)*m]
            Zt = Z[i*p:(i+1)*p, :]
            w_hat[i,: p, :] = dot(dot(Ht, inv(F[i])), v[i]) - dot(dot(Ht, K[i].T), r[i+1]) #e_hat:px1
            w_hat[i, p:p+m, :] = dot(dot(Q, Rt.T), r[i+1]) # n_hat: mx1
            r[i] = dot(dot(Zt.T, inv(F[i])), v[i]) + dot(L[i].T, r[i+1])
        self._w_hat = w_hat
        self._r = r

    def __forward_recursion_to_evaluate_alpha_hat(self, a, P0, T, R, Q):
        t, m, p, r = self._t, self._m, self._p, self._r
        alpha_hat = zeros((t + 1, m, 1))
        alpha_hat[0] = a[0] + dot(P0, r[0])
        for i in range(t):
            Tt = T[i * m:(i + 1) * m, i * m:(i + 1) * m]  # mxm
            Rt = R[i * m:(i + 1) * m, i * m:(i + 1) * m]
            alpha_hat[i + 1] = dot(Tt, alpha_hat[i]) + dot(dot(Rt, Q), dot(Rt.T, r[i]))
        self._alpha_hat = alpha_hat

