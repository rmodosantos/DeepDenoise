# Model Card
## Model Description

**Input:** Data corresponding to single channel images of any size. For example, a 2D array with size N x M, where N and M can take any positive integer value.

**Output:** Denoised data of the same size as the input.

**Model architecture:** The CNN architecture (DnCNN) has been already developed, showing good denoising performance. For details on the model design please refer to the original research article (Zhang K, Zuo W, Chen Y, Meng D, Zhang L. Beyond a gaussian denoiser: Residual learning of deep cnn for image denoising. *IEEE Transactions on Image Processing. 2017*) or to the detailed blog post at: https://sh-tsang.medium.com/review-dncnn-residual-learning-of-deep-cnn-image-denoising-super-resolution-jpeg-deblocking-cbf464b03130.

Briefly, the model consists in a deep convolutional neural network (20 layers) which implements residual learning using batch normalization to improve performance. The model implicitly learns to remove the latent image (or biological signal, in the context of this project) and outputs the noise (residual). The denoised output can then be obtained by subtracting the CNN output to the input data.

![DnCNN architecture](https://miro.medium.com/v2/resize:fit:2000/format:webp/1*Z0Qc0-ixlMKKs8EnPN3Z-Q.png)

## Performance
For a detailed assessment of model performance please check the notebook [Test_models](notebooks/Test_models.ipynb). Briefly, the two CNN models trained with different noise levels outperformed a gaussian smoothing algorithm. Notably, the CNN trained with less noised achieved the best overall performance, owing the less distortion of authentic neuromodulator dynamics extracted from fluorescence data. The better preservation of sharp features in the target data is clearly visible in a validation set with the noise level used for training (figure below).

[Validation instance](Validation_instance.png)

Remarkably, the qualitative differences between the CNN networks and gaussian smoothing still hold in a noisier test set, as assessed by the coherence of extracted neuromodulator dynamics with ground truth (figure below).

[Coherence](Coherence.png)

## Limitations
It is not clear how the CNN models generalize to other data types. The main concern on that matter are the domain-specific assumptions used to generate synthetic training instances.

## Trade-offs
The performance analysis suggests that the noise level used during training is an important hyperparameter to consider. It is apparent that CNNs trained with too much noise tend to ignore the sharp details on the target image and act more like a trivial smoothing algorithm. On the other hand, CNNs trained with low noise can better preserve sharp features but at the expense of lower noise cancelation. The lowest noise level tested seemed to provide a good trade-off between noise cancelation and signal preservation, but it might be possible to further fine-tune this parameter.

## Alternative architectures
If the preservation of sharp image features is a critical factor, other architectures explicitly designed to improve this aspect can be tested. Indeed, I have tested a few of them for this project, but unfortunatelly training and reaching good performance proved difficult. Such architectures included:
- [EFID: Edge-Focused Image Denoising Using a Convolutional Neural Network](https://ieeexplore.ieee.org/document/10025731)
- [Multilevel Edge Features Guided Network for Image Denoising](https://ieeexplore.ieee.org/document/9178433)
