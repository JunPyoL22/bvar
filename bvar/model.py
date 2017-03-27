
import numpy as np
from numpy.random import randn
from numpy.linalg import inv, cholesky
from bvar.base import BaseLinearRegression, BayesianModel, BasePrior, SetupForVAR
from bvar.sampling import Sampler
from bvar.utils import standardize, cholx, vec, DotDict

class BayesianLinearRegression(BayesianModel, Sampler):

    def __init__(self, *, n_iter=100, n_save=50, lag=0, y_type='multivariate',
                 sampling_method='Gibbs', prior_option={'Conjugate':'NormalWishart'},
                 alpha0=None, V0=None, V0_scale=10, v0=None, S0=None,
                 stability_check=False):
        '''
        :param n_iter: int, number of iteration
        :param n_save: int, number of time to save result
        :param lag: int, model lag, defult=0
        :param y_type: str, "univariate" or "multivariate"
        :param sampling_method: str, "Gibbs", "Metropolis_hasting"
        :param prior_option: dict, {"Conjugate":"Diffuse"} or {"Conjugate":"NormalWishart"},
                                   {"Conjugate":"NormalGamma"}
                                   {"NonConjugate":"Inden_NormalWishart"}
        :param alpha0: mean of Prior Normaldistribution
        :param V0: variance of Prior  Normaldistribution
        :param V0_scale: scale of variance of Prior Normaldistribution
        :param v0: degree of freedom of Prior  WishartDistribution
        :param S0: scale matrix of freedom of Prior  WishartDistribution
        :param stability_check: boolen, check sampled coeff is stable
        '''
        self.n_iter = n_iter
        self.n_save = n_save
        self.lag = lag
        self.y_type = y_type
        self.sampling_method = sampling_method
        self.prior_option = prior_option
        self.alpha0 = alpha0
        self.V0 = V0
        self.V0_scale = V0_scale
        self.v0 = v0
        self.S0 = S0
        self.stability_check = stability_check

    def estimate(self, Y: object, X: object) -> object:
        self.set_prior(Y, X).get_posterior_distribution()
        self.gibbs_sampling()
        return self

    def set_prior(self, Y, X):
        m, k = Y.shape[1], X.shape[1]

        if (self.y_type is 'univariate' and m != 1) or \
                (self.y_type is 'multivariate' and m == 1):
            raise ValueError('The dimension of Y is not {0}'.format(self.y_type))

        V0, scale = self.V0, self.V0_scale
        v0, S0 = self.v0, self.S0

        alpha0 = self.alpha0

        if k == 1:
            V0 = np.atleast_2d(list(self.V0))
            S0 = np.atleast_2d(list(self.S0))

        if list(self.prior_option.keys())[0] is 'Conjugate':
            if list(self.prior_option.values())[0] is 'Diffuse':
                self.prior = NaturalConjugatePrior(Y, X, 0*np.eye(k*m),
                                                   scale*np.eye(k),
                                                   0, 0.1*np.eye(m),
                                                   type='NonInformative')
                return self
            if 'Normal' in list(self.prior_option.values())[0]:
            # NormalWishart, NormalGamma
                self.prior = NaturalConjugatePrior(Y, X, alpha0,
                                                scale*V0, v0, S0,
                                                type='NormalWishart')
                return self

        # Non-conjugate
        else:
            return self

    def get_posterior_distribution(self):
        self.posterior = self.prior.get_posterior_distribution()
        

    def gibbs_sampling(self):
        mean, variance = self.posterior.normal_parameters.mean, \
                        self.posterior.normal_parameters.variance
        scale, dof = self.posterior.wishart_parameters.scale, \
                     self.posterior.wishart_parameters.dof

        km = mean.shape[0]
        m = scale.shape[0]

        if self.n_save >= 1:
            self.coef = np.empty((self.n_save, km))
            self.sigma = np.empty((self.n_save, m, m))

        for i in range(self.n_iter):

            coef = self.sampling_from_normal(mean, variance)
            if self.stability_check:
                '''should implement coef stability check later'''
                pass
            if self.y_type is 'multivariate':
                sigma = self.sampling_from_inverseWishart(scale, dof)
            else: #'''In case, univariate'''
                sigma = self.sampling_from_inverseGamma(scale, dof)

            # save
            if i >= self.n_save:
                self.coef[i-self.n_save:i-self.n_save+1, :] = coef[:, 0:1].T
                self.sigma[i-self.n_save:i-self.n_save+1, :, :] = sigma

            if self.n_save == 1:
                self.coef = coef[:, 0:1]
                self.sigma = sigma

        return self

class NaturalConjugatePrior(BasePrior, BaseLinearRegression):
    def __init__(self, Y, X, alpha0, V0, v0, S0, type='NomalWishart'):
        '''
        Natural Conjugate Prior
        alpha0: mean of prior Normal distribution
        V0: variance of prior Normal distribution
        v0: degree of freedom of prior Wishart Distribution
        S0: scale matrix of freedom of prior Wishart Distribution
        type: str, 'NormalWishart' or 'NonInformative'
        '''
        self.Y = Y
        self.X = X
        self.alpha0 = alpha0
        self.V0 = V0
        self.v0 = v0
        self.S0 = S0
        self.type = type

    def get_posterior_distribution(self):
        '''
        :param type: str, 'NormalWishart', 'NormalGamma'
        :return: mean: nparray vector, mean of posterior Noraml distribution
                 variance: nparray, variance covariance of posterior Normal distribution
        '''
        ols = BaseLinearRegression().fit(self.Y, self.X)
        if 'Normal' in self.type:
            self.normal_parameters = self._get_normal_posterior_parameters(ols.coef, ols.sigma)
            self.wishart_parameters = self._get_wishart_posterior_parameters(ols.coef, ols.sse)
        return self

    def _get_normal_posterior_parameters(self, alpha_ols, sigma):
        Y, X = self.Y, self.X
        alpha, V, k, m = self.alpha0, self.V0, X.shape[1], Y.shape[1]
        self.V_bar = inv(inv(V) + np.dot(X.T, X))
        self.A_bar = np.dot(V, (np.dot(inv(V), alpha)), np.dot(np.dot(X.T, X), alpha_ols))
        return DotDict({'mean': vec(self.A_bar),
                        'variance':np.kron(sigma, self.V_bar)})

    def _get_wishart_posterior_parameters(self, alpha_ols, sse):
        '''
        :param X:
        :param alpha_ols:
        :param sse:
        :return: scale: array, Scale matrix of posterior Wishart distribution
                 v: int, degree of freedom of posterior wishart distribution
        '''
        alpha, S, V, v = self.alpha0, self.S0, self.V0, self.v0
        Y = self.Y
        X = self.X
        self.S_bar = sse + \
                      S + \
                      np.dot(alpha_ols.T, np.dot(np.dot(X.T, X), alpha_ols)) + \
                      np.dot(alpha.T, np.dot(inv(V), alpha)) - \
                      np.dot(self.A_bar.T, np.dot(inv(V) + np.dot(X.T, X), self.A_bar))
        self.v_bar = Y.shape[0] + v
        return DotDict({'scale':self.S_bar, 'dof':self.v_bar})

class StateSpaceModel(object):
    '''
    State Space model
    (1) Observation Eq: y(t) = Z(t)*alpha(t) + e(t) e(t) ~ N(0,H(t))
    (2) Transition Eq: alpha(t) = T(t)*alpha(t) + R(t)n(t) n(t) ~ N(0,Q(t))
    '''
    def __init__(self,*,state0=None, state_var0=None,
                 smoother_mothod='DurbinKoopman'):
        self.state0 = state0
        self.state_var0 = state_var0
        self.smoother_method = smoother_mothod

    def estimate(self, y, state,
                 *, T=None, R=None, H=None, Q=None):
        pass

class FactorAugumentedVARX(BayesianLinearRegression):

    def __init__(self, n_iter=100, n_save=50, lag=1, var_lag=1, n_factor=3,
                 smoother_option='DurbinKoopman',tvp_option=False,
                 is_standardize=False):
        super().__init__(n_iter, n_save)
        self.smoother_option = smoother_option
        self.tvp_option = tvp_option
        self.n_factor = n_factor
        self.lag = lag
        self.var_lag = var_lag
        self.is_standardize = is_standardize

    def get_principle_component(self, Y):
        from utils import get_principle_component
        return get_principle_component(Y, self.n_factor)

    def get_factor_loadings(self, y, x):
        bayes_lreg = BayesianLinearRegression(n_iter=1, n_save=1, lag=0, y_type='univariate',
                                              prior_option={'Conjugate': 'NormalGamma'},
                                              alpha0=np.zeros((x.shape[1], 1)),
                                              V0=np.eye(x.shape[1]), V0_scale=1,
                                              v0=0, S0=0).estimate(y, x)
        return bayes_lreg.coef, bayes_lreg.sigma


    def estimate(self, Y, X, z):
        '''Assume Y, X has right form for VAR model to estimate
        must include or implement checking Y, X has right form to estimate'''
        t, m = Y.shape
        k = X.shape[1]

        if self.is_standardize is False:
            Y = standardize(Y)
            X = standardize(X)
            z = standardize(z)

        if self.n_factor != 0:
            self.factors, _ = self.get_principle_component(Y)


        state0 = None
        state0_var = None

        for ind in range(m):

            lag = self.lag
            y_i = Y[:, ind : ind + 1][lag:, :]
            z_i = z[:, ind : ind+1][lag:,:]
            y_i_lag = SetupForVAR(lag=lag, const=False).prepare(y_i).X
            z_lag = SetupForVAR(lag=lag, const=False).prepare(z_i).X
            x = np.c_[self.factors[lag:, :], y_i_lag, z[lag:, :], z_lag]

            coef_i, sigma_i = self.get_factor_loadings(y_i, x)


            for i in range(self.n_iter):
                pass

        #  state0 = pc_factor[0:1, :]

