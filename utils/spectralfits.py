import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import curve_fit
from scipy.signal import butter, filtfilt
import scipy.io as sio

# This file contains a set of functions used in the CNN test analysis

def fit_ACh_spectra(yspec, AChefnorm, Hb, HbO2):
    
    """Function to fit spectral features on a fluorescence signal.
    Inputs:
    yspec: Fluorescence data
    By default it assumes:
    ACh: Spectral profile of Acetylcholine effect in fluorescence
    Hb: deoxy-hemoglobin absorption spectrum
    HbO2: oxy-hemoglobin absorption spectrum 
    
    Output is an array where each column represent the time-series of intensity of a given feature:
    1 - Offset after mean normalization (inside exponent)
    2 - HbO2 intensity
    3 - Hb intensity
    4 - ACh intensity
    5 - Offset before mean normalization (outside exponent)"""
    
    # Calculate mean fluorescence for normalization
    neutralinvivo = np.mean(yspec, axis=0)
    # Normalize
    sig_n = yspec / neutralinvivo.reshape(1, -1)
    
    # Reshape inputs
    AChefnorm = AChefnorm.reshape(-1,152)
    HbInterp = Hb.reshape(-1,152)
    HbO2Interp = HbO2.reshape(-1,152)
    
    # Define fitting function
    def mF(x, m1, m2, m3, m4, m5):
        return 1. * np.exp(m1 - x[0,:] * m2 - x[1,:] * m3 + x[2,:] * m4) + m5 / neutralinvivo
    
    mfit = np.zeros((sig_n.shape[0], 5))  #fitting array
    Ly = sig_n.shape[0]
    
    # Fit the model
    for i in range(Ly):
        if np.sum(sig_n[i]) > 0:
            input_data = np.concatenate((HbO2Interp, HbInterp, AChefnorm),axis=0)
            popt, _ = curve_fit(mF, input_data, sig_n[i], p0=[0.1, 0.1, 0.1, 0.1, 1])
            mfit[i] = popt
    
    return mfit


def ButFilter(x, n, wn, flag):
    """
    Apply Butterworth filter to input signal 'x'.

    Parameters:
    - x: Input signal.
    - n: Filter order.
    - wn: Cutoff frequency or frequencies.
    - flag: Filter type ('low', 'high', 'band', 'stop').

    Returns:
    - y: Filtered output signal.
    """

    b, a = butter(n, wn, btype=flag, analog=False)
    y = filtfilt(b, a, x, axis=0)
    
    return y


def get_elements_in_periods(periods, A):
    """
    Extract elements from a vector that fall within specified intervals.

    Parameters:
    - periods (list of tuples or numpy array): List of intervals array specifying start and end of each period.
    - A (array-like): Input vector from which elements within intervals are to be extracted.

    Returns:
    - elements (numpy array): Array containing elements of 'A' within the specified intervals.
    - indices (numpy array): Boolean array marking the indices of 'A' that fall within the intervals.
    """
    result = []
    inds = []
    for start, end in periods:
        mask = np.logical_and(A >= start, A <= end)
        result.extend(A[mask])
        inds.extend(mask)
    return np.array(result),np.array(inds)


def denoise(data, model_path, denoised_path, network_depth=20, dev='cpu',save_file=True):
    """
    Denoise a fluorescence signal using the specified denoising network.

    Parameters:
    - data (ndarray): Input fluorescence signal.
    - model_path (str): Path to the trained denoising model checkpoint.
    - denoised_path (str): Path to save the denoised data (MATLAB .mat file).
    - network_depth (int): Depth of the denoising network.
    - dev (str): Device for computation ('cpu' or 'cuda' for GPU).
    - save_file (bool): Whether to save the denoised data as a file.

    Returns:
    - data_denoised (ndarray): Denoised fluorescence signal.
    """

    from os import remove
    
    # Move data to device
    dev=torch.device(dev)
    test_data = torch.tensor(data, dtype=float, device=dev)

    # Create CNN class
    class DnCNN(nn.Module):
        def __init__(self, depth=17, n_filters=64, kernel_size=3, n_channels=1,padding=1):
                """Pytorch implementation of DnCNN. Implementation followed the original paper [1]_. Authors original code can be
                found on `their Github Page
                <https://github.com/cszn/DnCNN/>`_.

                Notes
                -----
                This implementation is based on the following `Github page
                <https://github.com/SaoYan/DnCNN-PyTorch>`_.

                Parameters
                ----------
                depth : int
                    Number of fully convolutional layers in dncnn. In the original paper, the authors have used depth=17 for non-
                    blind denoising and depth=20 for blind denoising.
                n_filters : int
                    Number of filters on each convolutional layer.
                kernel_size : int tuple
                    2D Tuple specifying the size of the kernel window used to compute activations.
                n_channels : int
                    Number of image channels that the network processes (1 for grayscale, 3 for RGB)

                References
                ----------
                .. [1] Zhang K, Zuo W, Chen Y, Meng D, Zhang L. Beyond a gaussian denoiser: Residual learning of deep cnn
                    for image denoising. IEEE Transactions on Image Processing. 2017

                Example
                -------
                >>> from OpenDenoising.model.architectures.pytorch import DnCNN
                >>> dncnn_s = DnCNN(depth=17)
                >>> dncnn_b = DnCNN(depth=20)

                """
                super(DnCNN, self).__init__()
                layers = [
                    nn.Conv2d(in_channels=n_channels, out_channels=n_filters, kernel_size=kernel_size,
                            padding=padding, bias=False),
                    nn.ReLU(inplace=True)
                ]
                for _ in range(depth-2):
                    layers.append(nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=kernel_size,
                                            padding=padding, bias=False))
                    layers.append(nn.BatchNorm2d(n_filters))
                    layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Conv2d(in_channels=n_filters, out_channels=n_channels, kernel_size=kernel_size,
                                        padding=padding, bias=False))
                self.dncnn = nn.Sequential(*layers)


        def forward(self, x):
            out = self.dncnn(x)
            return out

    def reformat_data(data,window,SampRate=5):
        """
        Reformat the input data according to the denoising model's input requirements.

        Args:
            data (ndarray): Input fluorescence signal.
            window (int): Size of the data window.
            SampRate (int, optional): Sampling rate. Defaults to 5.

        Returns:
            datar (torch.Tensor): Reformatted data suitable for the denoising model.
            last_sample (torch.Tensor): The last segment of reformatted data.
        """
        window_sample = np.round(window*0.8)  #narrower denoising window to avoid border effects
        
        # Define the centers of data windows
        times = torch.arange(window*SampRate,len(data)-window*SampRate,window_sample*SampRate*2,dtype=int)
        
        # Pre-allocate output arrays
        datar = torch.zeros((times.shape[0]-1,1,2*window*SampRate+1,data.shape[1]),device=dev)
        last_sample = torch.zeros((1,1,len(data)-(times[-1]-int(window*SampRate)),data.shape[1]),device=dev)
        
        # Reformat equally-sized data
        for i,t in enumerate(times[:-1]):
            data_seg = data[t-window*SampRate:t+window*SampRate+1,:]
            datar[i,0,:,:] = data_seg
        
        # Reformat remaining data sample
        sample = data[times[-1]-int(window*SampRate):,:]
        last_sample[0,0,:,:] = sample
  
        return datar, last_sample
    

    # Denoise the recording
    def Denoise_recording(net,data, last_data, path, window, rescale, noise_scale=1, SampRate=5):
        """
        Apply denoising model to the input data and convert it to the original format.

        Args:
            net (nn.Module): Denoising neural network.
            data (torch.Tensor): Reformatted input data.
            last_data (torch.Tensor): Last segment of input data.
            path (str): Path to save the denoised data (MATLAB .mat file).
            window (int): Size of the data window.
            rescale (float): Rescaling factor.
            noise_scale (float, optional): Noise scaling factor. Defaults to 1.
            SampRate (int, optional): Sampling rate. Defaults to 5.
        """
        # Size of the border to exclude in each denoising window
        col_border = int(np.round(window-window*rescale)*SampRate)

        mat_data = []
        
        for i in range(len(data)):
            if i==0:
                # Denoise window
                tm_inst = data[i,:,:,:].unsqueeze(0)*noise_scale-net(data[i,:,:,:].unsqueeze(0)*noise_scale)
                tm_inst = tm_inst/noise_scale
                tm_inst = tm_inst[:,:,:-col_border-1,:]
                tm_inst = tm_inst.view(tm_inst.shape[2]*tm_inst.shape[0],tm_inst.shape[3])

                mat_data.append(tm_inst.cpu().detach().numpy())
                
            else:
                # Denoise window
                tm_inst = data[i,:,:,:].unsqueeze(0)*noise_scale-net(data[i,:,:,:].unsqueeze(0)*noise_scale)
                tm_inst = tm_inst/noise_scale
                tm_inst = tm_inst[:,:,col_border:-col_border-1,:]
                tm_inst = tm_inst.view(tm_inst.shape[2]*tm_inst.shape[0],tm_inst.shape[3])

                mat_data.append(tm_inst.cpu().detach().numpy())
                    
        # Denoise last data window
        tm_inst = last_data[:,:,col_border:,:]*noise_scale-net(last_data[:,:,col_border:,:]*noise_scale)
        tm_inst = tm_inst/noise_scale
        tm_inst = tm_inst.view(tm_inst.shape[2]*tm_inst.shape[0],tm_inst.shape[3])
        mat_data.append(tm_inst.cpu().detach().numpy())
        
        # Save to file
        sio.savemat(path, {"data": np.concatenate(mat_data, axis=0)})

    # Reformat data
    data_net,last_data_net = reformat_data(test_data,10)
    
    # load model checkpoint
    model_checkpoint = torch.load(
        model_path,map_location=dev)
    state_dict = model_checkpoint['model_state_dict']

    #create network object
    net = DnCNN(n_channels=1,depth=network_depth,kernel_size=3,padding=1)
    net.load_state_dict(state_dict)
    net.to(dev)

    #optimize scaling factor
    std_sample = []
    for sc in np.arange(0.2,1.5,0.025):
        data_seg = data_net[50,:,:,:].unsqueeze(0)*sc
        data_sample = data_seg.detach().cpu().numpy()-net(data_seg).detach().cpu().numpy()
        data_sample = np.apply_along_axis(lambda x: x/np.mean(x),0,np.squeeze(data_sample))
        std_sample.append(np.std(data_sample))

    scale = np.arange(0.2,1.5,0.025)[np.argmin(std_sample)]

    #Denoise
    Denoise_recording(net,data_net,last_data_net,denoised_path,
                  10,0.8, noise_scale=scale)
    
    #load denoised data
    data_denoised = sio.loadmat(denoised_path)['data']
    
    if not save_file:
        remove(denoised_path)

    return data_denoised