import numpy as np
from scipy.signal import welch
from scipy.optimize import minimize_scalar

def GetSn(Y, range_ff=[0.25, 0.5], method='logmexp'):
    # if len(Y.shape) == 1:
    #     Y = Y.reshape(-1, 1)
    # else:
    #     Y = Y.T
    print(np.shape(Y))
    psdx, ff = welch(Y, axis=0)
    indf = np.logical_and(ff >= range_ff[0], ff <= range_ff[1]).squeeze()
    
    print(np.shape(indf))
    if method == 'mean':
        sn = np.sqrt(np.mean(psdx[indf] / 2))
    elif method == 'median':
        sn = np.sqrt(np.median(psdx[indf] / 2))
    elif method == 'logmexp':
        sn = np.sqrt(np.exp(np.mean(np.log(psdx[indf] / 2))))
    else:
        print('Wrong method! Using logmexp instead.')
        sn = np.sqrt(np.exp(np.mean(np.log(psdx[indf] / 2))))
    
    return sn

def update_g(y, active_set, lam):
    len_active_set = active_set.shape[0]
    y = y.reshape(-1, 1)
    maxl = np.max(active_set[:, 3])
    c = np.zeros_like(y)
    
    def rss_g(g):
        h = np.exp(np.log(g) * np.arange(maxl+1))
        hh = np.cumsum(h * h)
        yp = y - lam * (1 - g)
        for ii in range(len_active_set):
            li = int(active_set[ii, 3])
            ti = int(active_set[ii, 2])
            idx = np.arange(ti, ti + li)
            tmp_v = np.maximum(np.sum(yp[idx] * h[:li]) / hh[li], 0)
            c[idx] = tmp_v * h[:li]
        res = y - c
        rss = np.sum(res * res)
        return rss
    
    result = minimize_scalar(rss_g, bounds=(0, 1), method='bounded')
    g = result.x
    h = np.exp(np.log(g) * np.arange(maxl+1))
    hh = np.cumsum(h * h)
    yp = y - lam * (1 - g)
    for m in range(len_active_set):
        li = int(active_set[m, 3])
        ti = int(active_set[m, 2])
        idx = np.arange(ti, ti + li)
        active_set[m, 0] = np.sum(yp[idx] * h[:li])
        active_set[m, 1] = hh[li]
    
    c, s, active_set = oasisAR1(y, g, lam, active_set=active_set)
    
    return c, active_set, g, s

def foopsi_oasisAR1(y, g, lam, optimize_b, optimize_g, decimate, maxIter):
    T = len(y)
    
    if decimate > 1:
        # To be done: Implement decimation
        pass
    
    if not optimize_b:
        b = 0
        solution, spks, active_set = oasisAR1(y, g, lam)
    else:
        # Optimize baseline here
        pass
    
    c = solution
    s = spks
    
    return c, s, b, g, active_set


def oasisAR1(y, g=None, lam=0, smin=0, active_set=None):
    y = y.reshape(-1, 1)
    T = len(y)
    
    if g is None:
        g = estimate_time_constant(y)
    
    if active_set is None:
        len_active_set = T
        active_set = np.column_stack([
            y - lam * (1 - g),
            np.ones((T, 1)),
            np.arange(0, T),
            np.ones((T, 1)),
            np.arange(-1, T-1),
            np.arange(1, T+1)
        ])
        
        print([y[-1,0] - lam, 1, T, 1, T - 1, np.nan])
        active_set[-1, :] = [y[-1,0] - lam, 1, T, 1, T - 1, np.nan]
        active_set[0, 4] = np.nan
    else:
        len_active_set = active_set.shape[0]
        active_set[:, 4] = np.hstack([np.nan, np.arange(0, len_active_set-1)])
        active_set[:, 5] = np.hstack([np.arange(1, len_active_set), np.nan])
    print(active_set[-3:,5])

    idx = np.ones(len_active_set, dtype=bool)
    
    ii = 0
    ii_next = int(active_set[ii, 5])
    while not np.isnan(ii_next):
        #print((1,ii_next))
        while (not np.isnan(ii_next)) and (
            active_set[ii_next, 0] / active_set[ii_next, 1] >=
            active_set[ii, 0] / active_set[ii, 1] * g ** active_set[ii, 3] + smin
        ):
            active_set[ii_next, 4] = ii
            ii = int(ii_next)
            #ii_next = int(active_set[ii, 5])
            if not np.isnan(active_set[ii, 5]):
                ii_next = int(active_set[ii, 5])
            else:
                ii_next = active_set[ii, 5]

            #print((2,ii_next))
            
        if np.isnan(ii_next):
            break
        
        # Merge pools
        active_set[ii, 0] += active_set[ii_next, 0] * (g ** active_set[ii, 3])
        active_set[ii, 1] += active_set[ii_next, 1] * (g ** (2 * active_set[ii, 3]))
        active_set[ii, 3] += active_set[ii_next, 3]
        active_set[ii, 5] = active_set[ii_next, 5]
        idx[ii_next] = False
        
        if not np.isnan(active_set[ii, 5]):
            ii_next = int(active_set[ii, 5])
        else:
            ii_next = active_set[ii, 5]
        
        #print((3,ii_next,ii))

        if not np.isnan(active_set[ii, 4]):
            ii_prev = int(active_set[ii, 4])
        else:
            ii_prev = active_set[ii, 4]
        
        # Backtrack until violations fixed
        while (not np.isnan(ii_prev)) and (
            active_set[ii, 0] / active_set[ii, 1] <
            active_set[ii_prev, 0] / active_set[ii_prev, 1] * g ** active_set[ii_prev, 3] + smin
        ):
            ii_next = ii
            ii = ii_prev
            active_set[ii, 0] += active_set[ii_next, 0] * (g ** active_set[ii, 3])
            active_set[ii, 1] += active_set[ii_next, 1] * (g ** (2 * active_set[ii, 3]))
            active_set[ii, 3] += active_set[ii_next, 3]
            active_set[ii, 5] = active_set[ii_next, 5]
            idx[ii_next] = False
            
            if not np.isnan(active_set[ii, 4]):
                ii_prev = int(active_set[ii, 4])
            else:
                ii_prev = active_set[ii, 4]

            if not np.isnan(active_set[ii, 5]):    
                ii_next = int(active_set[ii, 5])
            else:
                ii_next = active_set[ii, 5]
    
    active_set = active_set[idx, :]
    len_active_set = active_set.shape[0]
    
    print(len_active_set)

    c = np.zeros_like(y)
    s = np.zeros_like(y)
    
    for ii in range(len_active_set):
        t0 = int(active_set[ii, 2])
        tau = int(active_set[ii, 3])
        #print((t0,tau))
        #print(np.maximum(0, active_set[ii, 0] / active_set[ii, 1]) * (g ** np.arange(0, tau)))
        #print(np.maximum(0, active_set[ii, 0] / active_set[ii, 1]) * (g ** np.arange(0, tau)).reshape(-1,1))
        c[t0:(t0 + tau)] = np.maximum(0, active_set[ii, 0] / active_set[ii, 1]) * (g ** np.arange(0, tau)).reshape(-1,1)
    
    s[active_set[1:, 2].astype(int) - 1] = c[active_set[1:, 2].astype(int) - 1] - g * c[active_set[1:, 2].astype(int) - 2]
    
    return c, s, active_set

def estimate_time_constant(y):
    # Implementation of the estimate_time_constant function
    pass



def deconvolveCa_py(y, g, lam):
    y = y.reshape(-1, 1)
    win = 200
    sn = GetSn(y)
    
    c = y.copy()
    s = y.copy()
    
    c, s, b, g, active_set = foopsi_oasisAR1(y, g, lam, optimize_b=0, optimize_g=0, decimate=1, maxIter=10)
    
    options = {'b': b, 'g': g}
    
    return c, s, options

