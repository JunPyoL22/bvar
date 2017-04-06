
import numpy as np
from numpy.linalg import inv
from bvar.base import BaseLinearRegression, BayesianModel, BasePrior, SetupForVAR
from bvar.sampling import Sampler
from bvar.utils import standardize, cholx, vec, DotDict
from bvar.smoother import DurbinKoopmanSmoother

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
            self.prior_type = list(self.prior_option.values())[0].split('-')[1]
            self.prior = NaturalConjugatePrior(Y, X, alpha0,
                                                scale*V0, v0, S0,
                                                type=self.prior_type)
            return self

        # NonConjugate
        else:
            self.prior_type = list(self.prior_option.values())[0].split('-')[1]
            self.prior = NonConjugatePrior(Y, X, alpha0,
                                                scale*V0, v0, S0,
                                                type=self.prior_type)
            return self

    def get_posterior_distribution(self, *, coef_ols=None, sigma=None, sse_ols=None):
        self.posterior = self.prior.get_posterior_distribution(coef_ols=coef_ols,
                                                               sigma=sigma,
                                                               sse_ols=sse_ols)
        return self

    def get_conditional_posterior_distribution(self, *, coef_ols=None, drawed_value=None,
                                               dist_type=None):
        self.posterior = \
            self.prior.get_conditional_posterior_distribution(coef_ols=coef_ols,
                                                              drawed_value=drawed_value,
                                                              dist_type=dist_type)
        return self

    def gibbs_sampling(self):
        ols = self.fit(self.Y, self.X, method='ls')
        if self.prior_option_key is 'Conjugate':
            if self.prior_type is 'NonInformative':
                sigma = np.eye(self.Y.shape[1])
            elif self.prior_type is 'Informative':
                sigma = ols.sigma

            for i in range(self.n_iter):
                coef, sigma = self.sampling_from_posterior(coef_ols=ols.coef,
                                                            sigma=sigma,
                                                            sse_ols=ols.sse)
                self._save(coef, sigma, i)

        elif self.prior_option_key is 'NonConjugate':
            sigma = np.eye(self.m)

            for i in range(self.n_iter):
                coef, sigma = self.sampling_from_conditional_posterior(coef_ols=ols.coef,
                                                                        sigma=sigma)
                self._save(coef, sigma, i)
        return self

    def _save(self, coef, sigma, i):
        if i >= self.n_save:
            self.coef[i-self.n_save:i-self.n_save+1, :] = coef[:, 0:1].T
            self.sigma[i-self.n_save:i-self.n_save+1, :, :] = sigma

        if self.n_save == 1:
            self.coef = coef[:, 0:1]
            self.sigma = sigma

    def sampling_from_posterior(self, *, coef_ols=None, sigma=None,sse_ols=None):
        self.get_posterior_distribution(coef_ols=coef_ols,
                                        sigma=sigma,
                                        sse_ols=sse_ols)
        mean, variance = self.posterior.normal_parameters.mean, \
                         self.posterior.normal_parameters.variance
        scale, dof = self.posterior.wishart_parameters.scale, \
                     self.posterior.wishart_parameters.dof

        coef_drawed = self.sampling_from_normal(mean, variance)

        if self.stability_check:
            '''should implement coef stability check later'''
            pass
        if self.y_type is 'multivariate':
            sigma_drawed = self.sampling_from_inverseWishart(scale, dof)
        elif self.y_type is 'univariate':
            sigma_drawed = self.sampling_from_inverseGamma(scale, dof)
        return coef_drawed, sigma_drawed

    def sampling_from_conditional_posterior(self, *, coef_ols=None, sigma=None):
    
        self.get_conditional_posterior_distribution(coef_ols=coef_ols,
                                                    drawed_value=sigma,
                                                    dist_type='Normal')
        mean, variance = self.posterior.normal_parameters.mean, \
                         self.posterior.normal_parameters.variance

        coef_drawed = self.sampling_from_normal(mean, variance)

        self.get_conditional_posterior_distribution(coef_ols=coef_ols,
                                                    drawed_value=coef_drawed,
                                                    dist_type='Wishart')
        scale, dof = self.posterior.wishart_parameters.scale, \
                     self.posterior.wishart_parameters.dof

        if self.stability_check:
            '''should implement coef stability check later'''
            pass
        if self.y_type is 'multivariate':
            sigma_drawed = self.sampling_from_inverseWishart(scale, dof)
        elif self.y_type is 'univariate':
            sigma_drawed = self.sampling_from_inverseGamma(scale, dof)
        return coef_drawed, sigma_drawed
        
class NaturalConjugatePrior(BasePrior):

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

    def get_posterior_distribution(self, *, coef_ols=None,
                                   sigma=None, sse_ols=None):
        '''
        :param coef_ols: nparray, estimated by ols 
        :param sigma: nparray, drawed sigma
        :param sse_ols: nparray, computed sum of square error by ols 
        :return: mean: nparray vector, mean of posterior Noraml distribution
                 variance: nparray, variance covariance of posterior Normal distribution
        '''
        self.normal_parameters = self._get_normal_posterior_parameters(coef_ols, sigma)
        self.wishart_parameters = self._get_wishart_posterior_parameters(coef_ols, sse_ols)
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

class NonConjugatePrior(NaturalConjugatePrior):

    def __init__(self, Y, X, alpha0=None, V0=None,
                 v0=None, S0=None, prior_type='Informative'):
        '''
        Non Conjugate Prior
        :param alpha0: int or nparray, mean of prior independent Normal distribution
        :param int or nparray, variance of prior independent Normal distribution
        :param int or degree of freedom of prior independent Wishart Distribution
        :param int or nparray, scale matrix of freedom of prior independent Wishart Distribution
        :param prior_type: str, 'Informative' or 'NonInformative'
        '''
        super().__init__(self,Y, X, alpha0, V0, v0, S0, prior_type=prior_type)

    def get_conditional_posterior_distribution(self, *, coef_ols=None,
                                               drawed_value=None, dist_type=None):
        '''
        conditional posterior distribution
        :param coef_ols: nparray, estimated coefi by ols  
        :param drawed_value: nparray, drawed_value is drawed array from conditional Normal posterior distribution 
                                    or drawed array from conditional Wishart posterior distribution 
        :param dist_type: str, distribution type, "Normal", "Wishart"  
        :return: 
        '''
        if dist_type is 'Normal':
            self.normal_parameters = self._get_normal_posterior_parameters(coef_ols, drawed_value)
        elif dist_type is 'Wishart':
            self.wishart_parameters = self._get_wishart_posterior_parameters(drawed_value)
        return self

    def _get_normal_posterior_parameters(self, alpha_ols, sigma):
        '''
        :param alpha_ols: nparray, ols value of alpha 
        :param sigma: nparra, drawed sigma from Wishart or Gamma distribution
        :return: Dotdic, mean and variance of normal posterior
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
                             (Y - np.dot(X, reshaped_alpha))))
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

    def estimate(self, Y, X, z, w):
        '''Assume Y, X has right form for VAR model to estimate
        must include or implement checking Y, X has right form to estimate'''
        t, m = Y.shape

        if self.is_standardize is False:
            Y = standardize(Y)
            X = standardize(X)
            z = standardize(z)

        if self.n_factor != 0:
            factors, _ = self.get_principle_component(Y)

        self._W = self._get_W(w, m)
        self.gibbs_sampling(Y, z, factors)
        return self

    def _gibbs_sampling(self, Y, z, w, factors):
        m = Y.shape[1]
        lag = self.lag
        var_lag = self.var_lag

        for i in range(self.n_iter):
            self._A = np.empty((m, 2))
            self._B = np.empty((m, 2*lag))
            self._G = np.empty((m, m))
            self._H = np.empty((m, m*lag))
            self._Psi = np.empty((m, self.n_factor))
            self._r = np.empty((m, 1))

            for ind in range(Y.shape[1]):
                y_i = Y[:, ind: ind + 1][lag:, :]
                z_i = z[:, ind: ind + 1][lag:, :]
                y_i_lag = SetupForVAR(lag=lag, const=False).prepare(y_i).X
                z_i_lag = SetupForVAR(lag=lag, const=False).prepare(z_i).X

                x = self._get_factor_loading_regressor(y_i_lag, z_i, z_i_lag, factors)

                sigma0 = np.eye(y_i.shape[1])
                coef_i, sigma_i = self._get_factor_loadings(y_i, x, sigma0)
                self._hold_drawed_factor_loadings(coef_i, ind, m)
                self._r[ind:ind+1, 1] = sigma_i

            # Set STATE VAR model for update factors
            setup_var = SetupForVAR(lag=self.var_lag, const=False)
            t = setup_var.t
            state_Y = np.c_[setup_var.prepare(factors).Y, np.ones((t,1)), np.ones((t,1))]
            state_X = np.c_[setup_var.prepare(state_Y).X]
            H = np.diag(self._r[:, 0])


    def _get_W(self, w, m):
        W = np.empty((2*m,m))
        w_1 = np.zeros((1,m))
        for i in range(m):
            w_1[:,i:i+1] = 1
            w_2 = w[i,i+1,:]
            w_2[:,i:i+1] = 0
            W[i:(i+1)*2,:] = np.r_[w_1, w_2]
        return W

    def _get_factor_loadings(self, y, x, sigma0):
        sigma_i = sigma0
        ols = self.fit(y, x, method='ls')
        coef_i, sigma_i = \
            self._sampling_from_conditional_posterior(coef_ols=ols.coef,
                                                      sigam=sigma_i,
                                                      y_type='univariate')
        return coef_i, sigma_i

    def _hold_drawed_factor_loadings(self, coef, n, m):
        self._A[n:n+1,:] = np.c_[1,-1*coef[self.lag,:]]
        self._G[n:n + 1, :] = np.dot(self._A[n:n+1,:],
                                     self._W[n:n+1, :])
        for i in range(self.lag):
            self._B[n:n+1, i*2:(i+1)*2] = np.c_[coef[i, :],\
                                                coef[i+self.lag+1, :]]
            self._H[n:n+1, i*m:(i+1)*m] = np.dot(self._B[n:n+1, 2*i:2*(i+1)],
                                                 self._W[n:n+1, :])

    def gibbs_sampling(self, Y, X, z, sigma_i):

        lag = self.lag
        var_lag = self.var_lag

        for ind in range(Y.shape[1]):

            y_i = Y[:, ind: ind + 1][lag:, :]
            z_i = z[:, ind: ind + 1][lag:, :]
            y_i_lag = SetupForVAR(lag=lag, const=False).prepare(y_i).X
            z_i_lag = SetupForVAR(lag=lag, const=False).prepare(z_i).X

            self._set_state(factors, y_i_lag, z_i, z_i_lag)

            # set state0 and state0_var by VAR model lag(var_lag)
            if var_lag >= 2:
                temp0 = np.empty((1, 0))

                for i in range(var_lag - 1, 0, -1):
                    temp0 = np.append(temp0, self.state[i - 1:i, :])
                state0 = np.c_[temp0, np.zeros((1, self.state.shape[1]))]

            elif var_lag == 1:
                state0 = np.zeros((1, self.state.shape[1]))
            state0_var = np.eye(state0.shape[1])
            
            dk_smoother = DurbinKoopmanSmoother(state0, state0_var)

            self.set_prior(y_i, self.state)
            sigma_i = np.eye(y_i.shape[1])
            ols = self.fit(y_i, self.state, method='ls')

            # Data setup for the VAR transition equation of State Space Model
            setup_VAR = SetupForVAR(lag=var_lag, const=False).prepare()

            for i in range(self.n_iter):

                # sampling factor loads
                coef_i, sigma_i = \
                    self._sampling_from_conditional_posterior(coef_ols=ols.coef,
                                                              sigam=sigma_i,
                                                              y_type='univariate')

                # Set prior for VAR(state transition equation) sampling
                state_Y = setup_VAR.prepare(self.state).Y
                state_X = setup_VAR.prepare(self.state).X

                self.set_prior(state_Y, state_X)
                ols_var = self.fit(state_Y, state_X, method='ls')
                if i == 0: state_VAR_sigma = np.eye(state_Y.shape[1])

                state_VAR_coef, state_VAR_sigma = \
                    self._sampling_from_conditional_posterior(coef_ols=ols_var.coef,
                                                              sigma=state_VAR_sigma,
                                                              y_type='univariate')
                # set terms for state space model
                Z = np.c_[coef_i.T, np.zeros((1, self.state.shape[1]))]
                H = sigma_i
                m_var, k_var = state_Y.shape[1], state_X.shape[1]
                reshaped_state_VAR_coef = np.reshape(state_VAR_coef,(k_var, m_var))
                T = np.r_[reshaped_state_VAR_coef.T,
                          np.eye(m_var*(var_lag-1), k_var)]
                Q = np.c_[np.r_[state_VAR_sigma, np.zeros(state_VAR_sigma.shape)],
                          np.r_[np.zeros(state_VAR_sigma.shape), np.zeros(state_VAR_sigma.shape)]]
                R = np.eye(k_var)
                state = dk_smoother.smoothing(state_Y, Z=Z, T=T, R=R, H=H, Q=Q).state_tilda[:,:3]

                #update factors

                #reset state using updated factors
                self._set_state(y_i_lag, z_i, z_i_lag)
                pass

    def _get_factor_loading_regressor(self, factors, y_i_lag, z_i, z_i_lag):
        if self.lag == 0:
            regressor = np.c_[factors, z_i]
        elif self.lag > 0:
            regressor = np.c_[factors[self.lag:, :], y_i_lag, z_i, z_i_lag]
        return regressor


    def _set_state(self, factors, y_i_lag, z_i, z_i_lag):
        if self.lag == 0:
            self.state = np.c_[factors, z_i]
        elif self.lag > 0:
            self.state = np.c_[factors[self.lag:, :], y_i_lag, z_i, z_i_lag]
        return self

    def _sampling_from_conditional_posterior(self, *,
                                             coef_ols=None,
                                             sigma=None,
                                             y_type=None):
        self.y_type = y_type
        coef_drawed, sigma_drawed = self.sampling_from_conditional_posterior(coef_ols=coef_ols,
                                                                             sigma=sigma)
        return coef_drawed, sigma_drawed
