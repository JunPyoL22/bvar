import numpy as np
from numpy.linalg import inv
from base import BaseLinearRegression, BayesianModel, BasePrior, SetupForVAR
from sampling import Sampler
from utils import standardize, cholx, vec, DotDict
from smoother import DurbinKoopmanSmoother, CarterKohn

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
        :param V0: variance of Prior Normaldistribution
        :param V0_scale: scale parameter for variance of Prior Normaldistribution
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

    def estimate(self, Y, X, sigma=None):

        self._set_prior(Y, X)
        if self.n_save >= 1:
            self.coef = np.empty((self.n_save, self.k))
            self.sigma = np.empty((self.n_save, self.m, self.m))
            self.reshaped_coef = np.empty((self.n_save, self.k, self.m))

        self._gibbs_sampling(Y, X, sigma)
        return self

    def _set_prior(self, Y, X):

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
                                               prior_type=self.prior_type)
            return self

        # NonConjugate
        else:
            self.prior_type = list(self.prior_option.values())[0].split('-')[1]
            self.prior = NonConjugatePrior(Y, X, alpha0,
                                           scale*V0, v0, S0,
                                           prior_type=self.prior_type)
            return self

    def _get_posterior_distribution(self, *, coef_ols=None, sigma=None, sse_ols=None):
        self.posterior = self.prior.get_posterior_distribution(coef_ols=coef_ols,
                                                               sigma=sigma,
                                                               sse_ols=sse_ols)
        return self

    def _get_conditional_posterior_distribution(self, *, coef_ols=None, drawed_value=None,
                                                dist_type=None):
        self.posterior = \
            self.prior.get_conditional_posterior_distribution(coef_ols=coef_ols,
                                                              drawed_value=drawed_value,
                                                              dist_type=dist_type)
        return self

    def _gibbs_sampling(self, Y, X, sigma0):
        ols = self.fit(Y, X, method='ls')
        if sigma0 is None and self.prior_type is 'Informative':
            raise ValueError('When prior_type is Informative, sigma must be assigned')
        sigma = np.eye(self.m)
        if self.prior_type is 'Informative':
            sigma = sigma0

        for i in range(self.n_iter):
            if self.prior_option_key is 'Conjugate':
                coef, sigma = self._sampling_from_posterior(coef_ols=ols.coef,
                                                           sigma=sigma,
                                                           sse_ols=ols.sse)
            elif self.prior_option_key is 'NonConjugate':
                coef, sigma = self._sampling_from_conditional_posterior(coef_ols=ols.coef,
                                                                       sigma=sigma)
            self._save(coef, sigma, i)
        return self


    def _sampling_from_posterior(self, *, coef_ols=None, sigma=None, sse_ols=None):
        self._get_posterior_distribution(coef_ols=coef_ols,
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

    def _sampling_from_conditional_posterior(self, *, coef_ols=None, sigma=None):

        self._get_conditional_posterior_distribution(coef_ols=coef_ols,
                                                    drawed_value=sigma,
                                                    dist_type='Normal')
        mean, variance = self.posterior.normal_parameters.mean, \
                         self.posterior.normal_parameters.variance

        coef_drawed = self.sampling_from_normal(mean, variance)

        self._get_conditional_posterior_distribution(drawed_value=coef_drawed,
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
        
    def _save(self, coef, sigma, i):
        reshaped_coef = np.reshape(coef, (self.k, self.m), order='F')
        if i >= self.n_save:
            self.coef[i-self.n_save:i-self.n_save+1, :] = coef[:, 0:1].T
            self.sigma[i-self.n_save:i-self.n_save+1, :, :] = sigma
            self.reshaped_coef[i-self.n_save:i-self.n_save+1, :, :] = reshaped_coef

        if self.n_save == 1:
            self.coef = coef[:, 0:1]
            self.sigma = sigma
            self.reshaped_coef = reshaped_coef

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
        :param coef_ols: ndarray, estimated by ols 
        :param sigma: ndarray, drawed sigma
        :param sse_ols: ndarray, computed sum of square error by ols 
        :return: mean: ndarray vector, mean of posterior Noraml distribution
                 variance: ndarray, variance covariance of posterior Normal distribution
        '''
        self.normal_parameters = self._get_normal_posterior_parameters(coef_ols, sigma)
        self.wishart_parameters = self._get_wishart_posterior_parameters(coef_ols, sse_ols)
        return self

    def _get_normal_posterior_parameters(self, alpha_ols, sigma):
        Y, X = self.Y, self.X
        alpha, k, m = self.alpha0, X.shape[1], Y.shape[1]
        inv_V = self._set_inv_V()

        self.V_bar = inv(inv_V + np.dot(X.T, X))
        self.A_bar = np.dot(self.V_bar, (np.dot(inv_V, alpha) + np.dot(np.dot(X.T, X), alpha_ols)))
        return DotDict({'mean': vec(self.A_bar),
                        'variance': np.kron(sigma, self.V_bar)})

    def _get_wishart_posterior_parameters(self, alpha_ols, sse):
        '''
        :param alpha_ols: ndarray
        :param sse:
        :return: scale: array, Scale matrix of posterior Wishart distribution
                 v: int, degree of freedom of posterior wishart distribution
        '''
        alpha, S, v = self.alpha0, self.S0, self.v0
        inv_V = self._set_inv_V()
        Y = self.Y
        X = self.X
        self.S_bar = sse + \
                     S + \
                     np.dot(alpha_ols.T, np.dot(np.dot(X.T, X), alpha_ols)) + \
                     np.dot(alpha.T, np.dot(inv_V, alpha)) - \
                     np.dot(self.A_bar.T, np.dot(inv_V + np.dot(X.T, X), self.A_bar))
        self.v_bar = Y.shape[0] + v
        return DotDict({'scale':self.S_bar,
                        'dof':self.v_bar})

    def _set_inv_V(self):
        '''
        THis is for computaion of inverse array V, when V0 is zeros array
        '''
        if np.sum(self.V0) != 0:
            return inv(self.V0)
        return np.zeros(self.V0.shape)

class NonConjugatePrior(NaturalConjugatePrior):

    def __init__(self, Y, X, alpha0=None, V0=None,
                 v0=None, S0=None, prior_type='Informative'):
        '''
        Non Conjugate Prior
        :param alpha0: int or ndarray, mean of prior independent Normal distribution
        :param V0: int or ndarray, variance of prior independent Normal distribution
        :param v0: int or degree of freedom of prior independent Wishart Distribution
        :param S0: int or ndarray, scale matrix of freedom of prior independent Wishart Distribution
        :param prior_type: str, 'Informative' or 'NonInformative'
        '''
        super().__init__(Y, X, alpha0, V0, v0, S0, prior_type=prior_type)

    def get_conditional_posterior_distribution(self, *, coef_ols=None,
                                               drawed_value=None, dist_type=None):
        '''
        conditional posterior distribution
        :param coef_ols: ndarray, estimated coefi by ols
        :param drawed_value: ndarray, drawed_value is drawed array from conditional Normal posterior distribution
                             or drawed array from conditional Wishart posterior distribution
        :param dist_type: str, distribution type, "Normal", "Wishart"
        '''
        if dist_type is 'Normal':
            self.normal_parameters = self._get_normal_posterior_parameters(coef_ols, drawed_value)
        elif dist_type is 'Wishart':
            self.wishart_parameters = self._get_wishart_posterior_parameters(drawed_value)
        return self

    def _get_normal_posterior_parameters(self, alpha_ols, sigma):
        '''
        :param alpha_ols: ndarray, ols value of alpha
        :param sigma: ndarray, drawed sigma from Wishart or Gamma distribution
        :return: Dotdic, mean and variance of normal posterior
        '''
        alpha0, V0 = self.alpha0, self.V0
        X = self.X
        inv_V = self._set_inv_V()
        
        self.V = inv(inv_V + np.kron(inv(sigma), np.dot(X.T, X)))
        self.M = np.dot(self.V,
                       (np.dot(inv_V, alpha0) + np.dot(np.kron(inv(sigma), np.dot(X.T, X)), vec(alpha_ols))))
        return DotDict({'mean': self.M,
                        'variance': self.V})

    def _get_wishart_posterior_parameters(self, alpha):
        '''
        :param alpha: ndarray, drawed array from conditional Normal posterior distribution
        :return:
         - scale: ndarray, scale array of wishart distribution
         - dof: int, degree of freedom of wishart distribution
        '''
        S0, v0 = self.S0, self.v0
        Y, X = self.Y, self.X
        t, m = Y.shape
        k = X.shape[1]

        self.v = t + v0
        reshaped_alpha = np.reshape(alpha, (k, m), order='F') #alpha:k*mx1
        self.sigma = np.dot((Y - np.dot(X, reshaped_alpha)).T,
                            (Y - np.dot(X, reshaped_alpha)))
        return DotDict({'scale': self.sigma,
                        'dof': self.v})

class FactorAugumentedVARX(BayesianLinearRegression):

    def __init__(self, n_iter=100, n_save=50, lag=1, var_lag=1, n_factor=3,
                 smoother_option='DurbinKoopman', is_standardize=True):

        self.n_iter = n_iter
        self.n_save = n_save
        self.lag = lag
        self.var_lag = var_lag
        self.n_factor = n_factor
        self.smoother_option = smoother_option
        self.is_standardize = is_standardize

    def estimate(self, Y, z, w):
        '''Assume Y, X has right form for VAR model to estimate
           must include or implement checking dimension of Y, X for estimating'''
        t, m = Y.shape

        if self.is_standardize is False:
            Y = standardize(Y)
            z = standardize(z)

        if self.n_factor != 0:
            factors, _ = self._get_principle_component(Y)

        self._W = self._get_W(w, m)
        self._gibbs_sampling(Y, z, factors)
        return self

    def _get_principle_component(self, Y):
        from utils import get_principle_component
        return get_principle_component(Y, self.n_factor)

    def _gibbs_sampling(self, Y, z, factors):
        t, m = Y.shape
        n = self.n_factor
        lag = self.lag
        var_lag = self.var_lag
        r = np.ones((m, 1)) # variace of
        sigma = np.eye(n+1)

        for i in range(self.n_iter):
            self._A = np.empty((m, 2))
            self._B = np.empty((m, 2*lag))
            self._G = np.empty((m, m))
            self._H = np.empty((m, m*lag))
            self._Psi = np.empty((m, self.n_factor))
            self._F = np.empty((m, m*lag)) #(mxm)x!
            self._e = np.empty((t-lag, m))

            for ind in range(m):
                y_i = Y[:, ind: ind + 1]
                z_i = z[:, ind: ind + 1]
                y_i_lag = SetupForVAR(lag=lag, const=False).prepare(y_i).X
                z_i_lag = SetupForVAR(lag=lag, const=False).prepare(z_i).X

                x = self._get_factor_loading_regressor(factors, y_i_lag, z_i, z_i_lag)

                sigma0 = r[ind]

                me_model = BayesianLinearRegression(n_iter=1, n_save=1, lag=0, y_type='univariate',
                                                    prior_option={'NonConjugate':'Indep_NormalWishart-NonInformative'},
                                                    alpha0=np.zeros((x.shape[1], 1)),
                                                    V0=np.zeros(x.shape[1]), V0_scale=1,
                                                    v0=0, S0=0).estimate(y_i[lag:, :], x, sigma0)

                coef_i, reshaped_coef_i, sigma_i = me_model.coef, \
                                                   me_model.reshaped_coef, \
                                                   me_model.sigma
                self._hold_drawed_factor_loadings(coef_i, ind, m)
                r[ind:ind+1, :] = sigma_i
                self._e[:, ind:ind+1] = y_i[lag:, :] - np.dot(x, reshaped_coef_i)

            invG = inv(self._G)
            for i in range(1, lag+1):
                self._F[:, (i-1)*m:i*m] = np.dot(invG, self._H[:, (i-1)*m:i*m]) #mxm

            self._Gamma = np.dot(invG, self._Psi) #mxn_factor
            self._St = np.dot(invG, self._e.T).T #txm

            FX = dict()
            for i in range(1, lag+1):
                var_setup = SetupForVAR(lag=i, const=False).prepare(Y)
                FX[i] = np.dot(var_setup.X,
                               self._F[:, (i-1)*m:i*m]).sum(axis=1).reshape(var_setup.t, 1)

            # Set STATE VAR model to update factors dynamically
            state1 = factors[lag:, :]
            state2 = factors[lag:, :]
            for i in range(1, lag+1):
                state1 = np.append(state1, FX[i][lag-i:, :], axis=1)
                state2 = np.append(state2, np.ones((state2.shape[0], 1)), axis=1)

            var_setup = SetupForVAR(lag=var_lag, const=False).prepare(state1)
            state_var_Y = var_setup.Y
            state_var_X = var_setup.X
            alpha0 = np.zeros((state_var_Y.shape[1]*state_var_X.shape[1], 1))
            V0 = np.zeros((state_var_Y.shape[1]*state_var_X.shape[1],
                           state_var_Y.shape[1]*state_var_X.shape[1]))
            te_model = BayesianLinearRegression(n_iter=1, n_save=1, lag=0, y_type='multivariate',
                                                prior_option={'NonConjugate':'Indep_NormalWishart-NonInformative'},
                                                alpha0=alpha0, V0=V0, V0_scale=1,v0=0, S0=0).estimate(state_var_Y, state_var_X, sigma)

            coef, reshaped_coef, sigma = te_model.coef, te_model.reshaped_coef, te_model.sigma
            self._u = state_var_Y[:, :n] - \
                      np.dot(state_var_X, reshaped_coef)[:, :n]

            # Z_2 = np.empty((0, self._Gamma.shape[1]+lag))
            # for i in range(m):
            #     tiled_Gamma = np.tile(self._Gamma[i:i+1, :], (FX[lag].shape[0], 1)) #tx1
            #     Z_temp = np.empty((FX[lag].shape[0], 0))
            #     for j in range(1, lag+1):
            #         Z_temp = np.append(Z_temp, FX[j][lag-j:, :], axis=1)
            #         z2 = np.c_[tiled_Gamma, Z_temp]
            #     Z_2 = np.r_(Z_2, z2) #(mx(n+lag))

            Z, H, T, Q, R = self._get_state_space_model_parameters(coef, reshaped_coef, sigma, r,
                                                                   lag, var_lag, t)
            state0, \
            state0_var = self._get_initial_value_of_state(state1, var_lag)
            if self.smoother_option is 'DurbinKoopman':
                factors = DurbinKoopmanSmoother(state0, state0_var).smoothing(Y[lag:, :],
                                                                              Z=Z, T=T,
                                                                              R=R, H=H,
                                                                              Q=Q).state_tilda[:, :n]
            if self.smoother_option is 'CarterKohn':
                factors = CarterKohn(state0, state0_var).estimate(Y[lag:, :],
                                                                  Z=Z, T=T,
                                                                  R=R, H=H,
                                                                  Q=Q, s=n).drawed_state
    def _get_W(self, w, m):
        W = np.empty((2*m, m))
        w_1 = np.zeros((1, m))
        for i in range(m):
            w_1[:, i:i+1] = 1
            w_2 = w[i:i+1, :]
            w_2[:, i:i+1] = 0
            W[i*2:(i+1)*2, :] = np.r_[w_1, w_2]
        return W

    def _get_factor_loading_regressor(self, factors, y_i_lag, z_i, z_i_lag):
        if self.lag == 0:
            regressor = np.c_[factors, z_i]
        elif self.lag > 0:
            regressor = np.c_[factors[self.lag:, :], y_i_lag,
                              z_i[self.lag:, :], z_i_lag]
        return regressor

    def _hold_drawed_factor_loadings(self, coef, n, m):
        self._A[n:n+1, :] = np.c_[1, -1*coef[self.lag, :]]
        self._G[n:n+1, :] = np.dot(self._A[n:n+1, :],
                                   self._W[2*n:2*(n+1), :])
        self._Psi[n:n+1, :] = coef[-self.n_factor:, :].T

        for i in range(self.lag):
            self._B[n:n+1, i*2:(i+1)*2] = np.c_[coef[i, :],
                                                coef[i+self.lag+1, :]]
            self._H[n:n+1, i*m:(i+1)*m] = np.dot(self._B[n:n+1, 2*i:2*(i+1)],
                                                 self._W[2*n:2*(n+1), :])
        return self

    def _get_initial_value_of_state(self, state, lag):
        if lag >= 2:
            temp0 = np.empty((1, 0))
            for i in range(lag-1, 0, -1):
                temp0 = np.append(temp0, state[i-1:i, :], axis=1)
            state0 = np.c_[temp0, np.zeros((1, state.shape[1]))]

        elif lag == 1:
            state0 = np.zeros((1, state.shape[1]))
        state0_var = np.eye(state0.shape[1])
        return state0, state0_var

    def _get_state_space_model_parameters(self, coef, reshaped_coef, sigma,
                                          r, lag, var_lag, t):
        '''
        This function returns parameters on
        state space model with Non-timevaring parameters and Non-stochastic volatilty
        i.e Z, T, H, Q are constant all the time or not varying depends on time
        '''

        m_var, k_var = reshaped_coef.shape
        m, n = self._Gamma.shape
        Z = self._Gamma #mxn
        for i in range(1, lag + 1):
            Z = np.append(Z, np.ones((m, 1)), axis=1)
        # for indentifying factors
        Z[:n, :n] = np.eye(n)
        Z[:n, n:n+1] = np.zeros((n, 1))

        if var_lag == 1:
            T = reshaped_coef.T
            Q = sigma
        else:
            T = np.r_[reshaped_coef.T,
                      np.eye(m_var*(var_lag-1), k_var)]
            Q = np.zeros((k_var, k_var))
            Q[:m_var, :m_var] = sigma
        H = np.diag(r[:, 0]) #mxm
        R = np.eye(k_var)
        return np.tile(Z, (t-lag, 1)), np.tile(H, (t-lag, 1)),\
               np.tile(T, (t-lag, 1)), np.tile(Q, (t-lag, 1)),\
               np.tile(R, (t-lag, 1))

    def _sampling_parameters(self, y, x, sigma0):
        m, k = y.shape[1], x.shape[1]
        y_type = 'multivariate'
        if y.shape[1] == 1:
            y_type = 'univairate'
        sigma_i = sigma0
        ols = self.fit(y, x, method='ls')
        coef, sigma = \
            self._sampling_from_conditional_posterior(coef_ols=ols.coef,
                                                      sigma=sigma_i,
                                                      y_type=y_type)
        reshaped_coef = np.reshape(coef, (k, m), order='F')
        return coef, reshaped_coef, sigma

    def _set_state(self, factors, y_i_lag, z_i, z_i_lag):
        if self.lag == 0:
            self.state = np.c_[factors, z_i]
        elif self.lag > 0:
            self.state = np.c_[factors[self.lag:, :], y_i_lag,
                               z_i[self.lag:, :], z_i_lag]
        return self

    def _sampling_from_conditional_posterior(self, *,
                                             coef_ols=None,
                                             sigma=None,
                                             y_type=None):
        self.y_type = y_type
        coef_drawed, sigma_drawed = self.sampling_from_conditional_posterior(coef_ols=coef_ols,
                                                                             sigma=sigma)
        return coef_drawed, sigma_drawed
