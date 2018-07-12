import numpy as np
from numpy.linalg import inv
from numpy.random import randn
from bvar.utils import cholx


def draw_inverse_gamma(T0, D0, x):
    '''
    :param T0: initial degrees of freedom
    :param D0: initial scale parameter
    :param x: vector of innovations ,Tx1
    :return: sigma2: drawed
    '''
    T = x.shape[0]
    T1 = T0 + T
    D1 = D0 + np.dot(x.T, x) # 1x1
    z0 = randn(T1, 1) # T1x1
    z0z0 = np.dot(z0.T, z0) # 1x1
    sigma2 = D1/z0z0 # 1x1
    return sigma2 # 1x1

class Sampler(object):
    def sampling_from_normal(self, mean, variance):
        '''
        return: nparray, drawed array, kmx1 
        '''
        km = mean.shape[0]
        return mean + np.dot(cholx(variance).T, randn(km, 1))

    def sampling_from_inverseGamma(self, t0, v0, scale, dof):
        t1 = t0 + dof
        v1 = v0 + scale
        z0 = randn(t1, 1)
        return v1/np.dot(z0.T, z0)

    def sampling_from_inverseWishart(self, scale, dof):
        arr = np.dot(cholx(inv(scale)).T, randn(scale.shape[0], dof))
        return inv(np.dot(arr, arr.T))
        