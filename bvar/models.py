from numpy import dot, empty, append, kron, zeros, eye, ones, c_, \
                        r_, diag, atleast_2d, absolute, tile
from numpy.linalg import inv, cholesky, eigvals
from numpy.random import randn
from scipy.stats import multivariate_normal
from functools import wraps
from bvar.utils import lag, vec, cholx, DotDict, get_principle_component
from smoother import DurbinKoopmanSmoother
from bvar.sampling import draw_inverse_gamma

def argument_checker(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        for key, subkey in {'impulse_option': 'ihorizon', 'forecast_option': 'repfor'}.items():
            if kwargs.get(key) is True:
                if kwargs.get(subkey) is None:
                    raise ValueError('"{0}" must be integer when "{1}" is True'.format(subkey, key))
        return func(*args, **kwargs)
    return wrapper

class BayesianVAR(object):
    def __init__(self, Y, X, model_lag, n_trainning=None):
        self._Y = Y
        self._X = X
        self._p = model_lag # the number of lag on VAR
        self._t, self._m = Y.shape
        self._k = X.shape[1]
        self._Z = kron(eye(self._m),X)
        self._forecasted_Y = None
        self._predictive_likelihood = None
        self._drawed_alpha = None
        self._drawed_ALPHA = None
        self._drawed_sigma = None
        self._prior = None
        self._impulse_response = DotDict(dict())
        self._impulse_option = None
        self._forecast_option = None
        if n_trainning is None:
            self._n_trainning = self._t
        elif n_trainning != 0:
            self._n_trainning = n_trainning
        else:
            raise ValueError('n_trainning must greater than 0')


    @property
    def parameters(self):
        return DotDict({
            'alpha':self._drawed_alpha,
            'ALPHA':self._drawed_ALPHA,
            'SIGMA':self._drawed_sigma
        })

    @property
    def impulse_response(self):
        if self._impulse_option is True: return self._impulse_response
        else: raise ValueError('impluse option should be True')

    @property
    def forecast(self):
        if self._forecast_option is True: return self._forecasted_Y
        else: raise ValueError('forecast option should be True')


    def set_prior(self, scale=10, type='NormalWishart'):
        self._prior = {
            'NormalWishart': lambda : NaturalConjugatePriorNW(zeros((self._k, self._m)), scale*eye(self._k), self._m+1, eye(self._m)),
            'Diffuse': lambda : NonInformativePrior(self._k, self._m)
        }.get(type)
        if self._prior is None:
            raise KeyError('Can\'t find {0} Prior'.format(type))

    def __get_posterior_parameters(self, alpha_ols, sigma, sse):
        return self._prior().get_posterior_parameters(self._X, alpha_ols, sigma, sse, self._t)

    def __draw_alpha(self, mean, var):
        k, m = self._k, self._m
        return mean + dot(cholx(var).T, randn(k*m, 1))

    def __draw_sigma(self, S, dof):
        return self.iwpq(dof, inv(S))

    def __compute_impulse_response(self, phi, shock, horizon):
        '''
        Sck: square matrix of initial shock
        '''
        # neq: number of equations, nvar: number of variables
        nlag, neq, nvar = phi.shape
        responses = zeros((horizon, nvar, neq))
        responses[0, :, :] = shock.T
        for h in range(1, horizon):
            for ilag in range(min(nlag, h)):
                responses[h, :, :] = responses[h, :, :] + phi[ilag, :, :].dot(responses[h - ilag - 1, :, :])
        return responses

    def __save(self, nburn, alpha, ALPHA, sigma, i):
        self._drawed_alpha[i-nburn-1,:] = alpha[:, 0:1].T
        self._drawed_ALPHA[i-nburn-1,:,:] = ALPHA
        self._drawed_sigma[i-nburn-1,:,:] = sigma

    def get_ols_parameters(self):
        Y, X = self._Y[:self._n_trainning,:], self._X[:self._n_trainning,:]
        t, k = X.shape
        beta_ols = dot(inv(dot(X.T, X)), dot(X.T, Y))
        beta_ols_vec = vec(beta_ols)
        error = Y - dot(X, beta_ols)
        sigma_ols = (1 / (t - k + 1)) * dot(error.T, error)
        if self._n_trainning is not None:
            sigma_ols = (1 / t) * dot(error.T, error)
        return beta_ols, beta_ols_vec, sigma_ols, dot(error.T, error)

    @argument_checker
    def estimate(self, nsave=None, nburn=None, scale=None, prior_type='NormalWishart', \
                 impulse_option=False, ihorizon=None, forecast_option=False, repfor=None, stability_check=False):
        self._forecast_option, self._impulse_option = forecast_option, impulse_option
        self.set_prior(scale, type=prior_type)
        alpha_ols, alpha_vec_ols, sigma, sse = self.get_ols_parameters()
        k, m = self._k, self._m

        self._drawed_alpha = empty((nsave, k * m))
        self._drawed_ALPHA = empty((nsave, k, m))
        self._drawed_sigma = empty((nsave, m, m))

        self._forecasted_Y = empty((nsave*repfor,self._m))
        self._predictive_likelihood = empty((nsave, self._m))

        for i in range(m):
            self._impulse_response[i + 1] = zeros((nsave, m, ihorizon))

        for nloop in range(nsave + nburn):
            parameters = self.__get_posterior_parameters(alpha_ols, sigma, sse)
            if stability_check:
               alpha = self.__draw_alpha(parameters.Normal.mean, parameters.Normal.var)
               while self.is_var_coefficient_not_stable(alpha, m, self._p):
                   alpha = self.__draw_alpha(parameters.Normal.mean, parameters.Normal.var)
            else:
                alpha = self.__draw_alpha(parameters.Normal.mean, parameters.Normal.var)
            sigma = self.__draw_sigma(parameters.Wishart.S, parameters.Wishart.dof)
            ALPHA = alpha.reshape((k, m), order="F")

            if nloop > nburn:
                if forecast_option:
                    self.forecast(ALPHA, sigma, repfor, nloop, nburn)

                if impulse_option:
                    responses = self.get_impulse_response(ALPHA, sigma, ihorizon)
                    for i in range(m):
                        self._impulse_response[i + 1][nloop - nburn - 1, :, :] = responses[:, :, i].T

                self.__save(nsave, alpha, ALPHA, sigma, nloop)

    def get_impulse_response(self, ALPHA, sigma, ihor):
        p, m = self._p, self._m
        Bv = empty((p, m, m))
        for i in range(p):
            Bv[i,:,:] = ALPHA[1+(i*m):(i+1)*m+1,:]

        # shock : cholx(sigma)
        d = diag(diag(cholx(sigma)))
        shock = inv(d).dot(cholx(sigma))
        return self.__compute_impulse_response(Bv, shock, ihor)

    def forecast(self, ALPHA, sigma, repfor, nloop, nburn):
        t = self._t
        m, k, p = self._m, self._k, self._p
        forecasted_Y = empty((repfor, m))
        for i in range(repfor):
            forecast_for_X = c_[1, self._Y[t-1:, :], self._X[t-1:, 1:m*(p-1) + 1]]
            forecasted_Y[i,:] = forecast_for_X.dot(ALPHA) + randn(1, m).dot(cholx(sigma))

        self._forecasted_Y[((nloop-nburn)-1)*repfor:(nloop-nburn)*repfor-1,:] = forecasted_Y
        self._predictive_likelihood[nloop-nburn-1,:] = multivariate_normal.pdf(self._Y[t-1, :], mean=self._X[t-1,:].dot(ALPHA), cov=sigma)

    def iwpq(self, v, ixpx):
        k = ixpx.shape[0]
        z = zeros((v, k))
        mu = zeros((k, 1))
        for i in range(v):
            z[i] = mu.T + dot(cholesky(ixpx).T, randn(k, 1)).T  # 1 X k
        return inv(dot(z.T, z))

    def is_var_coefficient_not_stable(self, alpha, n, l):
        '''
            Input
             - coeffi: coefficients
             - n: number of endog variables(dimension of Y)
             - l: number of lags
            Output
             - if ee>1 not stable(True), else, stable(False), S=0
        '''
        FF = zeros((n * l, n * l))
        FF[n:n * l, :n * (l - 1)] = eye(n * (l - 1))
        FF[:n, :n * l] = alpha.reshape((n * l + 1, n))[:n * l, :n].T  # alpha.reshape((n*l+1,n)): 7*3
        ee = max(absolute(eigvals(FF)))
        return ee > 1


class BayesianVARforTVP(BayesianVAR):
    def __init__(self, Y, X, model_lag, n_trainning=None, smoothing_option='DurbinKoopman', tau=3.5e-04):
        super().__init__(Y, X, model_lag, n_trainning)
        self._smoothing_option = smoothing_option
        self._y = None
        self._Z = None
        self._tau = tau
        self._alpha0 = None
        self._Ht = None
        self._Q0 = None
        self._alpha0_var = None

    def get_initial_parameters(self):
        x0 = self._x[:self._n_trainning,:]
        alpha0, alpha0_vec, sigma0, sse0 = self.get_ols_parpameters()
        alpha0_var = kron(sigma0, inv(dot(x0.T,x0)))
        Q0 = dot(alpha0_var, self._n_trainning)*self._tau
        return alpha0_vec.T, sigma0, alpha0_var, Q0

    def ready_data_applying_DKSmoother(self):
        self._alpha0, H0, self._alpha0_var, self._Q0 = self.get_initial_parameters()
        transformer = TransformVARDataforApplyingSST(self._Y[self._t-self._n_trainning:,:], \
                self._X[self._t-self._n_trainning:, :], self._p)
        self._y, self._Z = transformer.data.Y, transformer.data.Z
        self._Ht = tile(H0, self._t).T

    @argument_checker
    def estimate(self, nsave=None, nburn=None, scale=None, prior_type='NormalWishart', \
                 impulse_option=False, ihorizon=None, forecast_option=False, repfor=None, stability_check=False):

        for nloop in range(nsave + nburn):

class TransformVARDataforApplyingSST:
    def __init__(self, Y, X, model_lag):
        '''
        :param Y: Endogenous variables in VAR
        :param X: should not have the constant term
        :param model_lag: lag of VAR model
        :param n_training: number of training sample
        '''
        self._t, self._m = Y.shape
        self._Y = Y
        self._X = X
        self._p = model_lag
        self._k = self._m * (self._m * self._p + 1)
        self._Z = empty((self._t * self._m, self._k))

    @property
    def data(self):
        return DotDict({'Y':self._get_Y(),
                        'Z': self._get_Z()})

    def _get_Y(self):
        if self._t < self._m:
            return self.Y
        return self.Y.T

    def _get_Z(self):
        '''
        :return: Z indicates State Space Observation equation
                Y(t) = Z(t)*alpha(t) + e(t) e(t) ~ N(0,H(t))
        '''
        for i in range(self._t):
            z_temp = eye(self._m)
            for j in range(self._p):
                X_i = self._X[i:i + 1, j * self._m:(j + 1) * self._m]
                z_temp = c_[z_temp, kron(eye(self._m), X_i)]
            try:
                self._Z[i * self._m:(i + 1) * self._m, :] = z_temp
            except Exception as e:
                print(e)
                raise ValueError('ValueError raised since X in this VAR model has the constant term')
        return self._Z

class FactorAugmentedVAR(BayesianVAR):
    def __init__(self, Y, X, Z, model_lag, n_factor=None, tvp_option = False):
        super().__init__(Y, X, model_lag)
        self._z = Z
        self._n_factor = n_factor
        self._pmat, _ = self.__apply_principle_component(self._Y, self._n_factor)
        self._error = empty((self._t, self._m))
        self._tvp_option = tvp_option

    def __apply_principle_component(self, n_factor):
        return get_principle_component(self.Y, n_factor)

    def __get_factor_loading(self, pmat, rmat):
        for i in range(self._m):
            y = self._Y[:, i:i+1][self._p:, :]
            x = c_[pmat, self._z]
            fl = self.__sampling_factor_loadings(y, x, rmat[i])
            self._error[:, i:i+1] = (y - dot(x, fl))[:, 0:1]

    def __sampling_factor_loadings(self, y, x, rmat_i):
        var = inv(inv(eye(x.shape[1])) + inv(atleast_2d(rmat_i)) * dot(x.T, x))  # KKxKK
        mean = dot(var, inv(atleast_2d(rmat_i)) * dot(x.T, y))  # KKx1
        return mean + dot(randn(1, var.shape[0]), cholx(var)).T

    def __sampling_variance_of_factor_loadings(self, dof, scale, error):
        return draw_inverse_gamma(dof, scale, error)

    def __get_variance_of_factor_loading(self):
        rmat = empty((1,self._m))
        for i in range(self._m):
            rmat[:, i:i+1] = self.__sampling_variance_of_factor_loadings(0, 0, self._error[:, i:i+1])
        return rmat

    def __sampling_var_coefficients(self):
        data_setup = DataSetupForVAR(c_[self._pmat,self._z], self._p, const=True)
        Y = data_setup.get_data.Y
        X = data_setup.get_data.X
        if self._tvp_option:
            self.__estimate_time_varing_parameters(Y, X[:,1:])

    def __estimate_time_varing_parameters(self, Y, X, stability_check=False):
        # X: does not have the constant term
        # m: dimension of Y(observation)
        # k: dimension of state
        # p: model lag
        t, m = Y.shape
        p = self._p
        k = m * (X.shape[1] + 1)  # X: txmp if X has constant term
        Z = empty((t * m, k))
        def generate_Z():
            nonlocal Z
            for i in range(t):
                z_temp = eye(m)
                for j in range(p):
                    X_i = X[i:i + 1, j * m:(j + 1) * m]
                    z_temp = c_[z_temp, kron(eye(m), X_i)]
                Z[i * m:(i + 1) * m, :] = z_temp
        generate_Z()
        b0 = zeros((p, 1))
        b_var0 = 4 * eye(p)
        F = eye(p)
        H0 = 0.01 * eye(m)
        Q0 = 40 * 3.5e-04 * eye(k)  # 3.5e-04=0.0035
        smoother = DurbinKoopmanSmoother()
        smoother.data =
        smoother.apply_smoother(m, k, t, Z, H0, Q0)

    def estimate(self):
        pass

class FactorAugmentedVARX(FactorAugmentedVAR):
    def __init__(self, Y, X, Z, W, model_lag, n_factor=None, TVP_option=None):
        super().__init__(Y, X, Z, model_lag, n_factor=n_factor, \
                         TVP_option=TVP_option)
        self._W = W
        self._A = empty((Y.shape[1], Z.shape[1]))
        self._B = empty((Y.shape[1], model_lag*2))

    def __get_factor_loading(self, pmat, rmat):
        for i in range(self._m):
            y = self._Y[:, i:i+1][self._p:, :]
            y_lag = DataSetupForVAR(self._Y[:, i:i+1], self._p, const=False).get_data.X
            z_lag = DataSetupForVAR(self._z, self._p, const=False).get_data.X
            x = c_[y_lag, self._z[self._p:], z_lag, pmat[self._p:]]
            fl = self.__sampling_factor_loadings(y, x, rmat[i])
            self._error[:, i:i+1] = (y - dot(x, fl))[:, 0:1]
            self.__store_factor_loadings_to_AB(fl.T)


class DataSetupForVAR(object):
    def __init__(self, Y, nlag, const=True, forecast_method=None, forecast_horizon=None):
        '''
        :param Y: endogenous datas
        :param nlag: int, the number of lag order
        :param const: Boolean, constant term in X
        :param forecast_method: string, 'Direct' or 'Iterate'
        :param forecast_horizon: int, the number of forecasting horizon
        '''
        self._Y = Y
        self._X = None
        self._t = None
        self._t_raw = Y.shape[0]
        self._m = None
        self._k = None
        self._nlag = nlag
        self._const = const
        if self.__is_parameter_on_forecasting(forecast_method, forecast_horizon):
            self._forecast_method = forecast_method
            self._forecast_horizon = forecast_horizon

    @property        
    def get_data(self):
        self.__prepare()
        return DotDict({'X':self._X, 'Y':self._Y, 'nobs':self._t})

    def __prepare(self):
        self._t = -self._nlag
        if self._forecast_method is not None:
            data = self.__get_Y_for_forecast()
            self._Y = data.Y1
            self._X = self.__get_X_for_VAR(data.Y2, const=self._const)
            if self._forecast_method is 'Direct':
                self._t += data.t - 1
            else:
                self._t += data.t - self._forecast_horizon
        else:
            self._X = self.__get_X_for_VAR(self._Y, const=self._const)
            self._Y = self._Y[self._nlag:,:]
            self._t += self._t_raw

    def __get_Y_for_forecast(self):
        return ForForecast(self._forecast_method, self._forecast_horizon).get_data_for_forecasting(self._Y, self._nlag)

    def __get_X_for_VAR(self, Y, const):
        '''
        :param const: include a constant column in X or Not
        '''
        t = Y.shape[0]
        p = self._nlag
        x = empty((t - p, 0))
        for i in range(1, p + 1):
            x_lag = lag(Y, L=i)[p:, :]
            x = append(x, x_lag, axis=1)
        if const:
            x = c_[ones((t - p, 1)), x]
        return x

    def __is_parameter_on_forecasting(self, forecast_method, forecast_horizon):
        if forecast_method is not None:
            if forecast_method not in ['Direct', 'Iterate']:
                raise ValueError('forecasting method must be one of "Direct", "Iterate"')
        if forecast_horizon is not None:
            if forecast_horizon < 0:
                raise ValueError('forecasting horizon("h") shouldn\'t be less than 0')
        return True

class ForForecast(object):
    def __init__(self, method_type, horizon):
        self._method_type = method_type
        self._h = horizon

    def get_data_for_forecasting(self, Y, nlag):
        '''
        Iterated forecasts:
            Y(t) = A0 + Y(t-1) x A1 + ... + Y(t-p) x Ap + e(t)
            so that in this case there are p lags of Y (from 1 to p).
        Direct h-step ahead foreacsts:
            Y(t+h) = A0 + Y(t) x A1 + ... + Y(t-p+1) x Ap + e(t+h)
            so that in this case there are also p lags of Y (from 0 to p-1).
        '''
        if self._method_type is 'Direct':
            return self.__direct(Y, nlag)
        return self.__iterate(Y, nlag)

    def __direct(self, Y, nlag):
        t = Y.shape[0] - self._h - 1
        Y1 = Y[self._h:, :][nlag:t, :]
        return DotDict( { 'Y1': Y1[:-self._h,:], 'Y2': Y[1:-self._h, :][:-self._h,:], 't': t } )

    def __iterate(self, Y, nlag):
        t = Y.shape[0]
        return DotDict( { 'Y1': Y[nlag:t, :][:-self._h, :], 'Y2': Y[:-self._h, :], 't': t } )

class Prior(object):
    def __init__(self):
        self._alpha = None
        self._sigma = None
        self._alpha_vec = None

    @property
    def alpha(self):
        return self._alpha

    @property
    def alpha_vec(self):
        return self._alpha_vec

    @property
    def sigma(self):
        return self._sigma

    def get_prior(self):
        raise NotImplementedError

    def iwpq(self, v, ixpx):
        k = ixpx.shape[0]
        z = zeros((v, k))
        mu = zeros((k, 1))
        for i in range(v):
            z[i] = mu.T + dot(cholesky(ixpx).T, randn(k, 1)).T  # 1 X k
        return inv(dot(z.T, z))


class NaturalConjugatePriorNW(Prior):
    def __init__(self, alpha, V, v, S): 
        '''
        initial hyper parameters: alpha,V,v,S
        '''
        self.hyperparameters = DotDict({
                'alpha':alpha,
                'V':V,
                'v':v,
                'S':S,
                'k':V.shape[0],
                'm':S.shape[0]
        })

        self._posterior = NormalWishartPosterior(self.hyperparameters)

    def get_posterior_parameters(self, X, alpha_ols, sigma, sse, t):
        mean, var = self._posterior.get_normal_posterior_parameters(X, alpha_ols, sigma)
        S, dof = self._posterior.get_wishart_posterior_parameters(X, alpha_ols, sse, t)
        return DotDict({'Normal':DotDict({'mean':mean,'var':var}),
                        'Wishart':DotDict({'S':S,'dof':dof})})

class NonInformativePrior(NaturalConjugatePriorNW):
    def __init__(self, k, m):
        self.alpha = zeros((k*m, 1))
        self.V = 0.01*eye(k)
        self.v = 0
        self.S = 0.01*eye(m)
        super(NonInformativePrior, self).__init__(self.alpha, self.V, self.v, self.S)
    
    # def get_prior(self,  X, alpha0, sigma0, sse, m, k, t):
    #     '''
    #     :param sigma0: initial value of sigma or sampled sigma
    #     :param alpha0: initial value of alpha or sampled alpha
    #     :param sse: sum of square error from ols
    #     :param m: dimension of Y
    #     :param k: dimension of X
    #     :param t: # of rows in Y
    #     '''
    #     V_post = kron(sigma0, inv(dot(X.T, X)))
    #     self._alpha_vec = alpha0 + dot(cholx(V_post).T, randn(k*m, 1))
    #     self._alpha = self._alpha_vec.reshape((k, m), order="F")
    #     self._sigma = self.iwpq(t-k, inv(sse))


class Posterior(object):

    def iwpq(self, v, ixpx):
        k = ixpx.shape[0]
        z = zeros((v, k))
        mu = zeros((k, 1))
        for i in range(v):
            z[i] = mu.T + dot(cholesky(ixpx).T, randn(k, 1)).T  # 1 X k
        return inv(dot(z.T, z))

class NormalWishartPosterior(object):
    def __init__(self, hypers):       
        self._A_bar = None
        self._V_bar = None
        self._S_bar = None
        self._v_bar = None
        self._hypers = hypers

    def get_normal_posterior_parameters(self, X, alpha_ols, sigma):
        alpha, V, k, m = self._hypers.alpha, self._hypers.V, self._hypers.k, self._hypers.m
        # A = alpha.reshape((k, m), order='F')
        self._V_bar = inv(inv(V) + dot(X.T,X))
        self._A_bar = dot(V, (dot(inv(V), alpha)), dot(dot(X.T, X), alpha_ols))
        return vec(self._A_bar), kron(sigma, self._V_bar)

    def get_wishart_posterior_parameters(self, X, alpha_ols, sse, t):
        alpha, S, V, v = self._hypers.alpha, self._hypers.S, self._hypers.V, self._hypers.v
        self._S_bar = sse + \
                      S + \
                      dot(alpha_ols.T, dot(dot(X.T, X), alpha_ols)) + \
                      dot(alpha.T, dot(inv(V), alpha)) - \
                      dot(self._A_bar.T, dot(inv(V) + dot(X.T, X), self._A_bar))
        self._v_bar = t + v
        return self._S_bar, self._v_bar

class NormalWishartPrior(Prior):
    def __init__(self):
        super(NormalWishartPrior, self).__init__()
        self._V_prior = None
        self._v_prior = None 
        self._A_vec_post = None
        self._S_prior = None

    def set_initial_value(self, m, k, s):
        '''
        :param m: dimension of Y
        :param k: dimension of X
        :param s: parameter of variance
        '''
        self._A_vec_prior = zeros((k*m,1))
        self._V_prior = s*eye(k)
        self._v_prior = m + 1
        self._S_prior = eye(m)

    def get_prior(self, X, alpha0, sse, m, k, t, s=10):
        self.set_initial_value(m, k, s)
        A_prior = zeros((k, m))
        V_post = inv(inv(self._V_prior)+dot(X.T,X))
        A_post = dot(V_post, dot(inv(self._V_prior), A_prior), \
                 dot(dot(X.T, X), alpha0))
        S_post = sse + self._S_prior + dot(alpha0.T, dot(dot(X.T, X), alpha0)) + \
                       dot(A_prior.T, dot(inv(self._V_prior), A_prior)) - \
                       dot(A_post.T, dot(inv(self._V_prior) + dot(X.T, X), A_post))
        v_post = self._v_prior + t
        self._alpha_vec = vec(A_post) + dot(cholx(kron(self._sigma, V_post)).T,randn(k * m, 1))
        self._alpha = self._alpha_vec.reshape((k, m), order='F')
        self._sigma = self.iwpq(v_post, inv(S_post))

class  IndependentNormalWishat(NormalWishartPrior):
        def __init__(self):
            super(IndependentNormalWishat, self).__init__()

        def get_prior(self, Y, X, Z, alpha0, sigma0, sse, m, k, t, s=10):
            self.set_initial_value(m, k, s)
            self._A_vec_prior = zeros((k*m,1))
            variance = kron(inv(sigma0), eye(t))
            V_post = inv(self._V_prior + dot(Z.T, variance.dot(Z)))
            a_post = V_post.dot(self._V_prior.dot(self._A_vec_prior) + dot(Z.T, variance.dot(vec(Y))))

            self._alpha_vec = a_post + dot(cholx(V_post).T, randn(k * m, 1))
            self._alpha = self._alpha_vec.reshape((k, m), order="F")

            v_post = t + self._v_prior
            error = Y - X.dot(self._alpha)
            S_post = self._S_prior + dot(error.T, error)
            self._sigma = self.iwpq(v_post, inv(S_post))


