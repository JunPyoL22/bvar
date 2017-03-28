
import numpy as np
from numpy.linalg import inv
from bvar.base import BaseLinearRegression, BayesianModel, BasePrior, SetupForVAR
from bvar.sampling import Sampler
from bvar.utils import standardize, cholx, vec, DotDict

class BayesianLinearRegression(BayesianModel, Sampler):

    def __init__(self, *, n_iter=100, n_save=50, lag=0, y_type='multivariate',
                 sampling_method='Gibbs', prior_option={'Conjugate':'NormalWishart-Informative'},
                 alpha0=None, V0=None, V0_scale=10, v0=None, S0=None,
                 stability_check=False):
        '''
        :param n_iter: int, number of iteration
        :param n_save: int, number of time to save result
        :param lag: int, model lag, defult=0
        :param y_type: str, "univariate" or "multivariate"
        :param sampling_method: str, "Gibbs", "Metropolis_hasting"
        :param prior_option: dict, {"Conjugate": "NormalWishart-Informative"},
                                   {"Conjugate": "NormalWishart-NonInformative"},
                                   {"Conjugate": "NormalGamma-Informative"},
                                   {"Conjugate": "NormalGamma-NonInformative"},
                                   {"NonConjugate": "Indep_NormalWishart-Informative"},
                                   {"NonConjugate": "Indep_NormalWishart-NonInformative"},
                                   {"NonConjugate": "Indep_NormalGamma-Informative"},
                                   {"NonConjugate": "Indep_NormalGamma-NonInformative"},
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

    def estimate(self, Y: object, X: object):
        
        self.set_prior(Y, X)
        if self.n_save >= 1:
            self.coef = np.empty((self.n_save, self.k))
            self.sigma = np.empty((self.n_save, self.m, self.m))

        self.gibbs_sampling()
        return self

    def set_prior(self, Y, X):

        self.m, self.k = Y.shape[1], X.shape[1]

        if (self.y_type is 'univariate' and self.m != 1) or \
                (self.y_type is 'multivariate' and self.m == 1):
            raise ValueError('The dimension of Y is not {0}'.format(self.y_type))

        V0, scale = self.V0, self.V0_scale
        v0, S0 = self.v0, self.S0

        alpha0 = self.alpha0

        if self.k == 1:
            V0 = np.atleast_2d(list(self.V0))
            S0 = np.atleast_2d(list(self.S0))

        self.prior_option_key = list(self.prior_option.keys())[0]
        if self.prior_option_key is 'Conjugate':
            prior_type = list(self.prior_option.values())[0].split('-')[1]
            self.prior = NaturalConjugatePrior(Y, X, alpha0,
                                                scale*V0, v0, S0,
                                                type=prior_type)
            return self

        # NonConjugate
        else:
            prior_type = list(self.prior_option.values())[0].split('-')[1]
            self.prior = NonConjugatePrior(Y, X, alpha0,
                                                scale*V0, v0, S0,
                                                type=prior_type)
            return self

    def get_posterior_distribution(self, alpha, sigma):
        self.posterior = self.prior.get_posterior_distribution()
        return self

    def get_conditional_posterior_distribution(self, value, dist_type=None):
        self.posterior = \
            self.prior.get_conditional_posterior_distribution(value, dist_type=dist_type)
        return self

    def gibbs_sampling(self):
        
        if self.prior_option_key is 'Conjugate':
            self.get_posterior_distribution()
            mean, variance = self.posterior.normal_parameters.mean, \
                            self.posterior.normal_parameters.variance
            scale, dof = self.posterior.wishart_parameters.scale, \
                        self.posterior.wishart_parameters.dof

            for i in range(self.n_iter):
                coef, sigma = self._sampling_from_posterior(mean, variance, scale, dof)
                self._save(coef, sigma, i)

        elif self.prior_option_key is 'NonConjugate':
            sigma = np.eye(self.m)

            for i in range(self.n_iter):
                coef, sigma = self._sampling_from_conditional_posterior(sigma=sigma)
                self._save(coef, sigma, i)
        return self

    def _save(self, coef, sigma, i):
        if i >= self.n_save:
            self.coef[i-self.n_save:i-self.n_save+1, :] = coef[:, 0:1].T
            self.sigma[i-self.n_save:i-self.n_save+1, :, :] = sigma
        if self.n_save == 1:
            self.coef = coef[:, 0:1]
            self.sigma = sigma

    def _sampling_from_posterior(self, mean, variance, scale, dof):

        coef = self.sampling_from_normal(mean, variance)

        if self.stability_check:
            '''should implement coef stability check later'''
            pass
        if self.y_type is 'multivariate':
            sigma = self.sampling_from_inverseWishart(scale, dof)
        elif self.y_type is 'univariate':
            sigma = self.sampling_from_inverseGamma(scale, dof)
        return coef, sigma

    def _sampling_from_conditional_posterior(self, *, sigma=None):
    
        self.get_conditional_posterior_distribution(sigma, dist_type='Normal')
        mean, variance = self.posterior.normal_parameters.mean, \
                         self.posterior.normal_parameters.variance

        coef = self.sampling_from_normal(mean, variance)

        self.get_conditional_posterior_distribution(coef, dist_type='Wishart')
        scale, dof = self.posterior.wishart_parameters.scale, \
                     self.posterior.wishart_parameters.dof

        if self.stability_check:
            '''should implement coef stability check later'''
            pass
        if self.y_type is 'multivariate':
            sigma = self.sampling_from_inverseWishart(scale, dof)
        elif self.y_type is 'univariate':
            sigma = self.sampling_from_inverseGamma(scale, dof)
        return coef, sigma
        
class NaturalConjugatePrior(BasePrior, BaseLinearRegression):

    def __init__(self, Y, X, alpha0, V0, v0, S0, prior_type='Informative'):
        '''
        Natural Conjugate Prior
        alpha0: mean of prior Normal distribution
        V0: variance of prior Normal distribution
        v0: degree of freedom of prior Wishart Distribution
        S0: scale matrix of freedom of prior Wishart Distribution
        prior_type: str, 'Informative' or 'NonInformative'
        '''
        self.Y = Y
        self.X = X
        self.V0 = V0
        self.alpha0 = alpha0
        self.v0 = v0
        self.S0 = S0
        self.prior_type = prior_type

    def get_posterior_distribution(self):
        '''
        :param type: str, 'Informative', 'NonInformative'
        :return: mean: nparray vector, mean of posterior Noraml distribution
                 variance: nparray, variance covariance of posterior Normal distribution
        '''
        ols = self.fit(self.Y, self.X, method='ls')

        if self.prior_type is 'NonInformative':
            sigma = np.eye(self.Y.shape[1])
        elif self.prior_type is 'Informative':
            sigma = ols.sigma

        self.normal_parameters = self._get_normal_posterior_parameters(ols.coef, sigma)
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
        :param alpha_ols: nparray 
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
        return DotDict({'scale':self.S_bar,
                        'dof':self.v_bar})

class NonConjugatePrior(NaturalConjugatePrior, BaseLinearRegression):

    def __init__(self, Y, X, alpha, sigma, alpha0=None, V0=None, v0=None, S0=None, prior_type='Informative'):
        '''
        Non Conjugate Prior
        alpha: nparray, drawed alpha from independent Normal distribution
        sigma: nparray, drawed sigma from independent Wishart distribution
        alpha0: int or nparray, mean of prior independent Normal distribution
        V0: int or nparray, variance of prior independent Normal distribution
        v0: int or degree of freedom of prior independent Wishart Distribution
        S0: int or nparray, scale matrix of freedom of prior independent Wishart Distribution
        prior_type: str, 'Informative' or 'NonInformative'
        '''
        super().__init__(self,Y, X, alpha0, V0, v0, S0, prior_type=prior_type)
        self.alpha = alpha
        self.sigma = sigma

    def get_conditional_posterior_distribution(self, drawed_value, dist_type=None):
        '''
        conditional posterior distribution
        :param drawed_value: nparray, drawed_value is drawed array from conditional Normal posterior distribution 
                                    or drawed array from conditional Wishart posterior distribution 
        :param dist_type: str, distribution type, "Normal", "Wishart"  
        :return: 
        '''
        if dist_type is 'Normal':
            self.normal_parameters = self._get_normal_posterior_parameters(ols.coef, drawed_value)
        elif dist_type is 'Wishart':
            self.wishart_parameters = self._get_wishart_posterior_parameters(drawed_value)
        return self

    def _get_normal_posterior_parameters(self, alpha_ols, sigma):
        '''
        :param alpha_ols: ols value of alpha 
        :param sigma: drawed sigma from Wishart or Gamma distribution
        :return: 
        '''
        alpha0, V0 = self.alpha0, self.V0
        X = self.X

        self.V = inv(inv(V0) + np.kron(inv(sigma), np.dot(X.T, X)))
        self.M = np.dot(self.V,
                       (np.dot(inv(V0), alpha0)+ np.kron(inv(sigma), np.dot(np.dot(X.T, X), alpha_ols))))
        return DotDict({'mean': self.M,
                        'variance': self.V})

    def _get_wishart_posterior_parameters(self, alpha):
        '''
        :param alpha: nparray, drawed array from conditional Normal posterior distribution 
        :return: 
         - scale: nparray, scale array of wishart distribution
         - dof: int, degree of freedom of wishart distribution
        '''
        S0, v0 = self.S0, self.v0
        Y, X = self.Y, self.X
        t, m = Y.shape
        k = X.shape
        self.v = t + v0

        reshaped_alpha = np.reshape(alpha, ((k, m))) #alpha:k*mx1
        self.sigma = np.dot((Y - np.dot(X, reshaped_alpha).T,
                             (Y - np.dot(X, reshaped_alpha))
        return DotDict({'scale': self.sigma,
                        'dof': self.v})

class FactorAugumentedVARX(BayesianLinearRegression):

    def __init__(self, n_iter=100, n_save=50, lag=1, var_lag=1, n_factor=3,
                 alpha0=None, V0=None, V0_scale=1, v0=None, S0=None,
                 smoother_option='DurbinKoopman',tvp_option=False,
                 is_standardize=False):

        super().__init__(n_iter=n_iter, n_save=n_save, lag=lag, y_type="univariate",
                         prior_option={'NonConjugate':'Indep_NormalWishart-NonInformative'},
                         alpha0=alpha0, V0=V0, V0_scale=V0_scale, S0=S0)
        self.smoother_option = smoother_option
        self.tvp_option = tvp_option
        self.n_factor = n_factor
        self.var_lag = var_lag
        self.is_standardize = is_standardize

    def get_principle_component(self, Y):
        from bvar.utils import get_principle_component
        return get_principle_component(Y, self.n_factor)

        # bayes_fl = BayesianLinearRegression(n_iter=1, n_save=1, lag=lag, y_type='univariate',
        #                                       prior_option={'NonConjugate': 'Indep_NormalWishart-NonInformative'},
        #                                       alpha0=np.zeros((x.shape[1], 1)),
        #                                       V0=np.eye(x.shape[1]), V0_scale=1,
        #                                       v0=0, S0=0).set_prior(y, x)

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

        self.gibbs_sampling(Y, X, z)
        return self

    def gibbs_sampling(self, Y, X, z, sigma_i):

        lag = self.lag
        var_lag = self.var_lag

        for ind in range(Y.shape[1]):

            y_i = Y[:, ind: ind + 1][lag:, :]
            z_i = z[:, ind: ind + 1][lag:, :]
            y_i_lag = SetupForVAR(lag=lag, const=False).prepare(y_i).X
            z_i_lag = SetupForVAR(lag=lag, const=False).prepare(z_i).X

            self._set_state(y_i_lag, z_i, z_i_lag)

            # set state0 by var_lag
            if var_lag >= 2:
                temp0 = np.empty((1, 0))

                for i in range(var_lag - 1, 0, -1):
                    temp0 = np.append(temp0, self.state[i - 1:i, :])
                state0 = np.c_[temp0, np.zeros((1, self.state.shape[1]))]

            elif var_lag == 1:
                state0 = np.zeros((1, self.state.shape[1]))

            state0_var = np.eye(state0.shape[1])

            self.set_prior(y_i, self.state)
            sigma_i = np.eye(y_i.shape[1])

            # Data setup for VAR transition equation of State Space Model
            setup_VAR = SetupForVAR(lag=var_lag, const=True)

            for i in range(self.n_iter):

                coef_i, sigma_i = self._sampling_from_conditional_posterior(sigma=sigma_i)

                # Set prior for VAR(state transition equation) sampling
                state_Y = setup_VAR.prepare(self.state).Y
                state_X = setup_VAR.prepare(self.state).X

                self.set_prior(state_Y, state_X)
                if i == 0: state_VAR_sigma = np.eye(state_Y.shape[1])

                state_VAR_coef, state_VAR_sigma = self._sampling_from_conditional_posterior(sigma=state_VAR_sigma)
                Z = np.c_[coef_i, np.zeros((1, self.state.shape[1]))]

                # state_VAR = BayesianLinearRegression(n_iter=1, n_save=1, lag=lag, y_type='multivariate',
                #                                          prior_option={'NonConjugate': 'Indep_NormalWishart-NonInformative'},
                #                                          alpha0=np.zeros((self.state.shape[1], 1)),
                #                                          V0=np.eye(self.state.shape[1]), V0_scale=1,
                #                                          v0=0, S0=0).estimate(state_y_i, state_x_i)

                #update factor

                #reset state
                self._set_state(y_i_lag, z_i, z_i_lag)
                pass

    def _set_state(self, y_i_lag, z_i, z_i_lag):
        if self.lag == 0:
            self.state = np.c_[self.factors, z_i]
        elif self.lag > 0:
            self.state = np.c_[self.factors[self.lag:, :], y_i_lag, z_i, z_i_lag]
        return self

    def _sampling_from_conditional_posterior(self, *, sigma=None):

        self.get_conditional_posterior_distribution(sigma, dist_type='Normal')
        mean, variance = self.posterior.normal_parameters.mean, \
                         self.posterior.normal_parameters.variance

        coef = self.sampling_from_normal(mean, variance)

        self.get_conditional_posterior_distribution(coef, dist_type='Wishart')
        scale, dof = self.posterior.wishart_parameters.scale, \
                     self.posterior.wishart_parameters.dof

        sigma = self.sampling_from_inverseGamma(scale, dof)
        return coef, sigma
