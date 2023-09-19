import numpy as np
from scipy.signal import welch
from scipy.optimize import minimize_scalar

# This file contains a set of functions used to deconvolute fluorescence ACh recordings.
# The code was adopted from the CalmAn repository at: https://github.com/flatironinstitute/CaImAn
# The scientific paper describing the aplpication of this deconvolution approach can be found at:
# https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005423#sec002


def oasisAR1(y, g, lam=0, smin=0, active_set=None):
    """
    Applies an Active Set method for a 1st order auto-regressive model to deconvolve a signal.
    
    Parameters:
    - y (ndarray): Input signal.
    - g (float): gamma parameter that models fluorescence impulse response (usually between 0.9 and 1
    )
    - lam (float): penalty parameter
    - smin (float): minimum spike size
    """
    y = y.reshape(-1, 1)
    T = len(y)
    
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
        
        active_set[-1, :] = [y[-1,0] - lam, 1, T, 1, T - 1, np.nan]
        active_set[0, 4] = np.nan
    else:
        len_active_set = active_set.shape[0]
        active_set[:, 4] = np.hstack([np.nan, np.arange(0, len_active_set-1)])
        active_set[:, 5] = np.hstack([np.arange(1, len_active_set), np.nan])

    idx = np.ones(len_active_set, dtype=bool)
    
    ii = 0
    ii_next = int(active_set[ii, 5])
    while not np.isnan(ii_next):
        
        while (not np.isnan(ii_next)) and (
            active_set[ii_next, 0] / active_set[ii_next, 1] >=
            active_set[ii, 0] / active_set[ii, 1] * g ** active_set[ii, 3] + smin
        ):
            active_set[ii_next, 4] = ii
            ii = int(ii_next)
        
            if not np.isnan(active_set[ii, 5]):
                ii_next = int(active_set[ii, 5])
            else:
                ii_next = active_set[ii, 5]

            
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

    c = np.zeros_like(y)
    s = np.zeros_like(y)
    
    for ii in range(len_active_set):
        t0 = int(active_set[ii, 2])
        tau = int(active_set[ii, 3])
        c[t0:(t0 + tau)] = np.maximum(0, active_set[ii, 0] / active_set[ii, 1]) * (g ** np.arange(0, tau)).reshape(-1,1)
    
    s[active_set[1:, 2].astype(int) - 1] = c[active_set[1:, 2].astype(int) - 1] - g * c[active_set[1:, 2].astype(int) - 2]
    
    return c, s, active_set


def deconvolveCa_py(y, g, lam):
    y = y.reshape(-1, 1)
    
    c, s, _ = oasisAR1(y, g, lam)
    
    return c, s

