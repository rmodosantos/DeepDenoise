import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import curve_fit
from scipy.signal import butter, filtfilt
import scipy.io as sio

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




def ButFilter(x, n, wn, flag):
    """Function that filters the signal"""
    b, a = butter(n, wn, btype=flag, analog=False)
    y = filtfilt(b, a, x, axis=0)
    
    return y

def get_elements_in_periods(periods, A):
    result = []
    inds = []
    for start, end in periods:
        mask = np.logical_and(A >= start, A <= end)
        result.extend(A[mask])
        inds.extend(mask)
    return np.array(result),np.array(inds)

def denoise(data, model_path, denoised_path, network_depth=20, dev='cpu',save_file=True):
    """Denoise a fluorescence signal using the specified denoising network"""

    from os import remove

    dev=torch.device(dev)
    test_data = torch.tensor(data, dtype=float, device=dev)


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
        """Function used to reformat the data according to the denoising model input"""
        window_sample = np.round(window*0.8)
        times = torch.arange(window*SampRate,len(data)-window*SampRate,window_sample*SampRate*2,dtype=int)
        datar = torch.zeros((times.shape[0]-1,1,2*window*SampRate+1,data.shape[1]),device=dev)
        datar_original = torch.zeros((times.shape[0]-1,1,2*window*SampRate+1,data.shape[1]),device=dev)
        #mean_data = torch.zeros((times.shape[0],data.shape[1]))
        last_sample = torch.zeros((1,1,len(data)-(times[-1]-int(window*SampRate)),data.shape[1]),device=dev)
        last_sample_original = torch.zeros((1,1,len(data)-(times[-1]-int(window*SampRate)),data.shape[1]),device=dev)

        for i,t in enumerate(times[:-1]):
            data_seg = data[t-window*SampRate:t+window*SampRate+1,:]
            datar[i,0,:,:] = data_seg
              
        sample = data[times[-1]-int(window*SampRate):,:]
        last_sample[0,0,:,:] = sample
  
        return datar, last_sample
    

    # Denoise the recording
    def Denoise_recording(net,data, last_data, path, window, rescale, noise_scale=1, SampRate=5):
        """Apply denoising model and convert data to original format"""

        col_border = int(np.round(window-window*rescale)*SampRate)

        mat_data = []
        
        for i in range(len(data)):
            if i==0:

            #with h5py.File(path, 'w') as f:

                tm_inst = data[i,:,:,:].unsqueeze(0)*noise_scale-net(data[i,:,:,:].unsqueeze(0)*noise_scale)
                tm_inst = tm_inst/noise_scale
                tm_inst = tm_inst[:,:,:-col_border-1,:]
                tm_inst = tm_inst.view(tm_inst.shape[2]*tm_inst.shape[0],tm_inst.shape[3])

                #dset = f.create_dataset('data', data=tm_inst.cpu().detach().numpy(), dtype='f',maxshape=(None,152),chunks=(1,152))  # 'f' stands for float32
                mat_data.append(tm_inst.cpu().detach().numpy())
                
            else:
            
            
                tm_inst = data[i,:,:,:].unsqueeze(0)*noise_scale-net(data[i,:,:,:].unsqueeze(0)*noise_scale)
                tm_inst = tm_inst/noise_scale
                tm_inst = tm_inst[:,:,col_border:-col_border-1,:]
                tm_inst = tm_inst.view(tm_inst.shape[2]*tm_inst.shape[0],tm_inst.shape[3])

               
                mat_data.append(tm_inst.cpu().detach().numpy())
                    
        
        tm_inst = last_data[:,:,col_border:,:]*noise_scale-net(last_data[:,:,col_border:,:]*noise_scale)
        tm_inst = tm_inst/noise_scale
        tm_inst = tm_inst.view(tm_inst.shape[2]*tm_inst.shape[0],tm_inst.shape[3])
        mat_data.append(tm_inst.cpu().detach().numpy())
            
        sio.savemat(path, {"data": np.concatenate(mat_data, axis=0)})

    

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