import numpy as np

def gewekes_P(y, a=.3, b=.4, c=.3, sl=0.05):
    """
     - y: sampled params or datas that is nxd array
     - a,b,c: divide y into 3 parts using these(default:a=.3,b=.4,c=.3)
     - sl: significance level, default=0.05
    """
    y, n, d = check_validity_array(y) # y:nxd array
    na = int(n*a)
    nc = int(n*c)
    gewekesp = np.empty(d)
    for i in range(d):
        mean_a = np.mean(y[:na,i])
        mean_c = np.mean(y[nc:,i])
        std_a = np.std(y[:na,i])
        std_c = np.std(y[nc:,i])
        cd = (mean_a - mean_c)/((std_a/np.sqrt(na))+(std_c/np.sqrt(nc))) # convergence diagonostic
        gewekesp[i] = 2*(1-np.norm.cdf(abs(cd), loc=0, scale=1))
    sls = sl*np.ones(d)
    # null hypo: the sampled distribution converges to posterior distributioon
    res = gewekesp > sls
    return res, gewekesp

def check_validity_array(arr):
    arr = np.atleast_2d(arr)
    n, d = arr.shape
    if n < d:
        arr = arr.T
        n, d = arr.shape
    return arr, n, d

def inefficiency_factor(params, k, B=200):
    """
    This has a purpose to get the inefficiency_factor of sampled params by means of MCMC
    - params: sampled parameters
    - B: Constant
    - k: maximum lag of autocorrelation to compute
    Output:
    - ie_factors: npx1 array
    """
    nd, np = params.shape # nd: the number of draw, np:the number of parameters
    if nd < np:
        params = params.T
        nd, np = params.shape
    # auto_coeffi = np.zeros((k,np))
    auto_coeffi = np.empty((k,0))

    for i in range(np):
        # auto_coeffi[:,i] = autocorrelation_coefficients(params[:,i],k)
        auto_coeffi = np.append(auto_coeffi,autocorrelation_coefficients(params[:, i], k), axis=1)

    ie_factors = np.zeros((np,1))
    for i in range(np):
        sum_rho = 0
        for j in range(k):
            sum_rho = sum_rho + parzenkernel(j/k)*auto_coeffi[j,i]
        ie_factors[i] = 1+2*sum_rho
    return ie_factors

def parzenkernel(z):
    if z>=0 and z<=.5:
        value = 1 - 6*(z**2) + 6*(z**3)
    elif z>=.5 and z<=1:
        value = 2*((1-z)**3)
    else:
        value = 0
    return value

def autocorrelation_coefficients(y,k):
    """
    This function is to get the k th order autocorrelation coefficients on y
    - y: time-series data
    - k: k-th order
    Output:
    - auto_coeff: k th order autocorrelation coefficients 2dmension matirx; kx1
    """
    y = np.atleast_2d(y)
    t, d = y.shape
    if t < d:
        y = y.T
        t, d = y.shape
    if d > 1:
        raise ValueError("time seris data must be a vector")
    if t < k:
        raise ValueError("the number of rows of time series data should be bigger than k th order")
    if t > 1 and d <= 1:
        y = y[:,0]

    auto_coeff = np.ones((k,1))
    for i in range(1,k+1):
        Y = y[i:]
        X = y[:-i]
        auto_coeff[i-1] = np.corrcoef(Y,X)[0,1]
    return auto_coeff
