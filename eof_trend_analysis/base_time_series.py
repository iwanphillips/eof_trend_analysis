import packages; import importlib; importlib.reload(packages)
from packages import *

def line(x, a, b):
    return a + b*x 

# get d

def calc_rms(x, scale, order):

    shape = (x.shape[0]//scale, scale)
    X = np.lib.stride_tricks.as_strided(x,shape=shape)

    scale_ax = np.arange(scale)
    rms = np.zeros(X.shape[0])
    for e, xcut in enumerate(X):
        coeff = np.polyfit(scale_ax, xcut, order) 
        xfit = np.polyval(coeff, scale_ax)

        rms[e] = np.sqrt(np.mean((xcut-xfit)**2))
    return rms

def dfa_standard(x, order, scale_lim=[5,9], scale_dens=0.25):

    y = np.cumsum(x - np.mean(x))
    scales = (2**np.arange(scale_lim[0], scale_lim[1], scale_dens)).astype(np.int)
    fluct = np.zeros(len(scales))

    for e, sc in enumerate(scales):
        fluct[e] = np.sqrt(np.mean(calc_rms(y, sc, order)**2))
        
    coeff = np.polyfit(np.log2(scales), np.log2(fluct), 1)
    #plt.plot(np.log2(scales), np.log2(fluct), 'o')
    #plt.plot(scales, fluct, 'o')

    return scales, fluct, coeff 

def peak(x, c):
    return np.exp(-np.power(x - c, 2) / 16.0)

def lin_interp(x, y, i, half):
    return x[i] + (x[i+1] - x[i]) * ((half - y[i]) / (y[i+1] - y[i]))

def half_max_x(x, y):
    half = max(y)/2.0
    signs = np.sign(np.add(y, -half))
    zero_crossings = (signs[0:-2] != signs[1:-1])
    zero_crossings_i = np.where(zero_crossings)[0]

    if len(zero_crossings_i) == 0:
        return [x[np.argmax(y)], x[np.argmax(y)]]
    if len(zero_crossings_i) == 1:
        return [lin_interp(x, y, zero_crossings_i[0], half),
                np.max(x)]
    
    return [lin_interp(x, y, zero_crossings_i[0], half),
            lin_interp(x, y, zero_crossings_i[1], half)]

def generate_individual_ensembles_ordered_fixed(d2, limits, lpower = 0, epower = 1, min_points = 10):
    
    idx = np.where(np.logical_and(d2[:,0] >= limits[0], d2[:,0] <= limits[1]))[0]
    
    lhs_arr, rhs_arr, slope = [], [], []
    error, weights, offset  = [], [], []
    
    count = 0
    for lhs_idx in range(1, idx.shape[0] - min_points): 
        
        for rhs_idx in range(lhs_idx+min_points, idx.shape[0]):
            
            lhs = idx[lhs_idx]
            rhs = idx[rhs_idx]

            [p, e, _, _, _] = np.polyfit(d2[:,0][lhs:rhs], 
                     d2[:,1][lhs:rhs], deg=1, full=True)
            
            fit_error = 1e-10 + np.sqrt((e[0]) / (rhs_idx - lhs_idx - 1))
            fit_length = np.sqrt((d2[:,0][rhs] - d2[:,0][lhs]) ** 2 + (p[0]*(d2[:,0][rhs] - d2[:,0][lhs])) ** 2) 
            
            lhs_arr.append(lhs); rhs_arr.append(rhs); slope.append(p[0])
            offset.append(p[1]); error.append(fit_error)
            weights.append((fit_length ** lpower)/(fit_error ** epower))
            
            count += 1
        
    return [np.array(lhs_arr), np.array(rhs_arr), np.array(slope), 
            np.array(error), np.array(weights), np.array(offset)] 

def dfa_bradley(x, order, scale_lim=[4,10], scale_dens=0.2, P = 2, Q = 2, plot=False):
    # Previously I was thinking the default should be scale_lim=[5,10]

    y = np.cumsum(x - np.mean(x))
    scales = (2**np.arange(scale_lim[0], scale_lim[1],
                        scale_dens)).astype(int)
    fluct = np.zeros(len(scales))

    for e, sc in enumerate(scales):
        fluct[e] = np.sqrt(np.mean(calc_rms(y, sc, order)**2))
   
    region = np.array( [[ np.log2(i) , np.log2(j) ] for i,j in zip(scales, fluct)] )

    bounds = [region[0,0], region[-1,0]]
    [lhs_arr, rhs_arr, slope, error, weights, o] = generate_individual_ensembles_ordered_fixed(region, bounds, P, Q)
    idx = ~np.isnan(slope)
    slope = slope[idx]
    weights = weights[idx]
    kernel = gaussian_kde(slope, weights=weights)
    positions = np.linspace(0, 2, num=5000)  # num = 5000 to produce less discrete values. Else num = 500 
    mean = positions[np.argmax(kernel(positions))]  
    
    lower_x, upper_x = half_max_x(positions,kernel(positions))
    error = [mean-lower_x, upper_x-mean]    # np.std(fwhm)

    if plot==True: 
        return scales, fluct, mean, error, kernel, positions, weights, slope
    if plot=='fits':
        return region, scales, fluct, mean, error, lhs_arr, rhs_arr, slope, error, weights, o[idx]
    else:
        return scales, fluct, mean, error 

    
# get phi 

#@jit(nopython=True, parallel=True)
def grlet_diff(data_, memory_, d_, delta_t_=1.0):
    """Fractional differencing of data with first-order 
    approximation of Gruenwald-Letnikov derivative"""
    N_ = int(memory_/delta_t_)
    differenced_data = np.zeros((len(data_) - N_))

    diff_coefficients = [1.0]*N_
    for k in range(1, N_):
        diff_coefficients[k] = diff_coefficients[k-1] * ((k-1.0) - d_) / k

    for i in range(len(differenced_data)):
        summe = 0
        for j in range(N_):
            summe += diff_coefficients[j] * data_[N_-1+i-j]

        differenced_data[i] = delta_t_**(-d_) * summe

    return differenced_data


# arfima

def __ma_model(
    params: list[float],
    n_points: int,
    *,
    noise_std: float = 1,
    noise_alpha: float = 2,
) -> list[float]:

    ma_order = len(params)
    if noise_alpha == 2:
        noise = norm.rvs(scale=noise_std, size=(n_points + ma_order))
    else:
        noise = levy_stable.rvs(
            noise_alpha, 0, scale=noise_std, size=(n_points + ma_order)
        )

    if ma_order == 0:
        return noise
    ma_coeffs = np.append([1], params)
    ma_series = np.zeros(n_points)
    for idx in range(ma_order, n_points + ma_order):
        take_idx = np.arange(idx, idx - ma_order - 1, -1).astype(int)
        ma_series[idx - ma_order] = np.dot(ma_coeffs, noise[take_idx])
    return ma_series[ma_order:]

def __arma_model(params: list[float], noise: list[float]) -> list[float]:

    ar_order = len(params)
    if ar_order == 0:
        return noise
    n_points = len(noise)
    arma_series = np.zeros(n_points + ar_order)
    for idx in np.arange(ar_order, len(arma_series)):
        take_idx = np.arange(idx - 1, idx - ar_order - 1, -1).astype(int)
        arma_series[idx] = np.dot(params, arma_series[take_idx]) + noise[idx - ar_order]
    return arma_series[ar_order:]


def __frac_diff(x: list[float], d: float) -> list[float]:

    def next_pow2(n):
        return (n - 1).bit_length()

    n_points = len(x)
    fft_len = 2 ** next_pow2(2 * n_points - 1)
    prod_ids = np.arange(1, n_points)
    frac_diff_coefs = np.append([1], np.cumprod((prod_ids - d - 1) / prod_ids))
    dx = ifft(fft(x, fft_len) * fft(frac_diff_coefs, fft_len))
    return np.real(dx[0:n_points])


def arfima(
    ar_params: list[float],
    d: float,
    ma_params: list[float],
    n_points: int,
    *,
    noise_std: float = 1,
    noise_alpha: float = 2,
    warmup: int = 0,
) -> list[float]:
  
    ma_series = __ma_model(
        ma_params, n_points + warmup, noise_std=noise_std, noise_alpha=noise_alpha
    )
    frac_ma = __frac_diff(ma_series, -d)
    series = __arma_model(ar_params, frac_ma)
    return series[-n_points:]


# get p value

def asymp(a, d, T):
    
    f = ((1+a)/((1-a)*(2*hyp2f1(1,d,1-d,a) - 1)))*((36*(1-2*d)*gamma(1-d))/(d*(1+2*d)*(3+2*d)*gamma(d)))
    var = f*(T**(2*d-3))
    
    return(var)

def ztest(x, value):
   
    x = np.asarray(x)
    mean = np.mean(x)
    std_diff = np.std(x) 
   
    zstat = (mean - value) / std_diff
    pvalue = stats.norm.cdf(zstat)
   
    return(zstat, pvalue)

def ttest(x, value):
    
    x = np.asarray(x)
    mean = np.mean(x)
    std_diff = np.std(x) 
   
    tstat = abs(mean - value) / std_diff
    pvalue = 1 - stats.t.cdf(tstat, df=10000)
   
    return(tstat, pvalue)

def ttest_an(a, d, var, value, N=25567): 
    
    mean = 0
    std_diff = sqrt(asymp(a, d, N)*var)
   
    tstat = abs(mean - value) / std_diff
    pvalue = 1 - stats.t.cdf(tstat, df=10000)
    
    vtr = asymp(a, d, N)*var
   
    return(pvalue, vtr)

#session.evaluate('deriv[a_, d_] = (72 (-1 + a^2) (1 - 2 d) Gamma[1 - d])/((1 - a)^2 d (1 + 2 d) (3 + 2 d)^2 Gamma[d] (-1 + 2 Hypergeometric2F1[1, d, 1 - d, a])) + (72 (-1 + a^2) (1 - 2 d) Gamma[1 - d])/((1 - a)^2 d (1 + 2 d)^2 (3 + 2 d) Gamma[d] (-1 + 2 Hypergeometric2F1[1, d, 1 - d, a])) + (36 (-1 + a^2) (1 - 2 d) Gamma[1 - d])/((1 - a)^2 d^2 (1 + 2 d) (3 + 2 d) Gamma[d] (-1 + 2 Hypergeometric2F1[1, d, 1 - d, a])) + (72 (-1 + a^2) Gamma[1 - d])/((1 - a)^2 d (1 + 2 d) (3 + 2 d) Gamma[d] (-1 + 2 Hypergeometric2F1[1, d, 1 - d, a])) + (36 (-1 + a^2) (1 - 2 d) Gamma[1 - d] PolyGamma[0, 1 - d])/((1 - a)^2 d (1 + 2 d) (3 + 2 d) Gamma[d] (-1 + 2 Hypergeometric2F1[1, d, 1 - d, a])) + (36 (-1 + a^2) (1 - 2 d) Gamma[1 - d] PolyGamma[0, d])/((1 - a)^2 d (1 + 2 d) (3 + 2 d) Gamma[d] (-1 + 2 Hypergeometric2F1[1, d, 1 - d, a])) + (72 (-1 + a^2) (1 - 2 d) Gamma[1 - d] (-\!\(\*SuperscriptBox[\(Hypergeometric2F1\), TagBox[RowBox[{"(", RowBox[{"0", ",", "0", ",", "1", ",", "0"}], ")"}],Derivative],MultilineFunction->None]\)[1, d, 1 - d, a] + \!\(\*SuperscriptBox[\(Hypergeometric2F1\), TagBox[RowBox[{"(", RowBox[{"0", ",", "1", ",", "0", ",", "0"}], ")"}],Derivative],MultilineFunction->None]\)[1, d, 1 - d, a]))/((1 - a)^2 d (1 + 2 d) (3 + 2 d) Gamma[d] (-1 + 2 Hypergeometric2F1[1, d, 1 - d, a])^2)')

def ttest_an_derr(a, d, var, value, del_d, N=25567): 
    
    mean = 0
    std_diff = sqrt(asymp(a, d, N)*var)
   
    tstat = abs(mean - value) / std_diff
    pvalue = 1 - stats.t.cdf(tstat, df=10000)
    
    vtr = []
    #vtr.append(var*(N**(2*d-3))*np.sqrt(asymp(a, d, 1)**2 + (del_d[0]**2)*(session.evaluate('deriv['+str(a)+','+str(d)+']')**2)))
    #vtr.append(var*(N**(2*d-3))*np.sqrt(asymp(a, d, 1)**2 + (del_d[1]**2)*(session.evaluate('deriv['+str(a)+','+str(d)+']')**2)))
   
    return(pvalue, vtr)
    
def hist(y, ar, d, r_trend): 
   
    x = np.arange(len(y))
    
    trends = []
    X_l = sm.add_constant(x)
    var = np.var(y)
   
    for s in range(1000):
        series = arfima(ar, d, [], len(y))
        v = (np.sqrt(var)/np.sqrt(np.var(series)))
        series = [i*v for i in series]
        res_l = sm.OLS(series, X_l).fit()
   
        trends.append(res_l.params[1])
       
    tstat, tp = ttest(trends, r_trend)
    vtr = np.var(trends)
   
    return(tp, vtr)    

def p_val(y, ar=1, an=True, d_b=0, order=2): 
    
    x = np.arange(len(y))

    scales, fluct, coeff, d_err = dfa_bradley(y, order, scale_lim=[6,11]) 
    # run again with below code 
    #scales, fluct, coeff = dfa_standard(x, order, scale_lim=[5,11], scale_dens=0.25)
    #d_err = 0
    #print(coeff)
    
    d = float(coeff-0.5); print(d) 
    if d_b =='upp': 
        d, d_err = d + d_err[1], np.nan
    if d_b =='low': 
        d, d_err = d - d_err[0], np.nan
    
    diff = grlet_diff(y, 1095, d) # y, 10, d
    x1 = diff[:-1]; x2 = diff[1:]
    popt, pcov = scipy.optimize.curve_fit(line, x1, x2) 
    
    if ar == 0: ar, a = [], 0
    if ar == 1: ar, a = [popt[1]], popt[1]
    
    var = np.var(y)
    if var < 0: print("var < 0, var = ", var)

    popt, pcov = scipy.optimize.curve_fit(line, x, y) 
    alph, r_trend = popt 
    
    if an==False: 
        tp, vtr = hist(y, ar, d, r_trend)
    elif an=='derr':
        tp, vtr = ttest_an_derr(a, d, var, r_trend, d_err, N=len(y)) 
    else: 
        tp, vtr = ttest_an(a, d, var, r_trend, N=len(y)) 
    
    return(d, a, r_trend, var, tp, vtr, d_err) 


def plot_trend(temp, xlab=[1950, 1980, 2010]):

    f, ax = plt.subplots(1, 1, figsize=(4, 3)) 
    
    x = np.arange(len(temp))
    X_l = sm.add_constant(x)
    res = sm.OLS(temp, X_l).fit()
    
    pred_ols = res.get_prediction()
    iv_l = pred_ols.summary_frame()["obs_ci_lower"]
    iv_u = pred_ols.summary_frame()["obs_ci_upper"]
    
    ax.set_ylabel('T in K')
    ax.set_xlabel('Year')
    ax.set_xticks([0, 30*365.25, 60*365.25]) 
    ax.set_xticklabels(xlab) 
    ax.set_yticks([-10,0,10])

    ax.plot(temp, "-", linewidth=0.2, label="data")
    ax.axhline(0, color='k', ls='--', alpha=0.5)
    ax.plot(x, res.fittedvalues, "r-", linewidth=2, label="OLS")
    ax.plot(x, iv_u, "r--")
    ax.plot(x, iv_l, "r--")
    ax.legend(loc="best", ncol=2, fontsize=14)


def detrend(df):
    
    df.loc[(df.Temp < -100),'Temp']=np.nan
    yl = df['Temp'].to_numpy()
    
    y366 = df.groupby(['month', 'day'], as_index=False)['Temp'].mean()
    
    y365 = y366.drop(y366[(y366.month == 2) & (y366.day == 29)].index)
    y365 = y365['Temp'].tolist()
    
    y366 = y366['Temp'].tolist()
    y4 = y365 + y365 + y366 + y365
    ym = np.resize(y4, len(yl))
    ymf = savgol_filter(ym, 51, 3)
    
    yl2 = np.subtract(yl,ym)
    
    return yl2

