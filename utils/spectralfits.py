import numpy as np
from scipy.optimize import curve_fit

def fit_ACh_spectra(yspec, AChefnorm, Hb, HbO2):
    # Assumes the standard wavelength vector from 350 to 1000 nm from spectrometer data
    
    neutralinvivo = np.mean(yspec, axis=0)
    #sig_n = yspec[:, sw] / neutralinvivo.reshape(1, -1)
    sig_n = yspec / neutralinvivo.reshape(1, -1)
    

    AChefnorm = AChefnorm.reshape(-1,152)
    HbInterp = Hb.reshape(-1,152)
    HbO2Interp = HbO2.reshape(-1,152)
    
    def mF(x, m1, m2, m3, m4, m5):
        return 1. * np.exp(m1 - x[0,:] * m2 - x[1,:] * m3 + x[2,:] * m4) + m5 / neutralinvivo
    
    yfity = sig_n
    
    mfit = np.zeros((yfity.shape[0], 5))
    Ly = yfity.shape[0]
    
    # Fit the model
    for i in range(Ly):
        if np.sum(yfity[i]) > 0:
            #print(np.shape(np.concatenate((HbO2Interp, HbInterp, AChefnorm),axis=1))
            input_data = np.concatenate((HbO2Interp, HbInterp, AChefnorm),axis=0)
            #print(np.shape(input_data))
            popt, _ = curve_fit(mF, input_data, yfity[i], p0=[0.1, 0.1, 0.1, 0.1, 1])
            mfit[i] = popt
    
    return mfit