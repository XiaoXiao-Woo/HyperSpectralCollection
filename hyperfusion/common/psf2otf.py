import numpy as np
from numpy import pad as zero_pad
import torch


def psf2otf(psf, shape):
    """
    Convert point-spread function to optical transfer function.

    Compute the Fast Fourier Transform (FFT) of the point-spread
    function (PSF) array and creates the optical transfer function (OTF)
    array that is not influenced by the PSF off-centering.
    By default, the OTF array is the same size as the PSF array.

    To ensure that the OTF is not altered due to PSF off-centering, PSF2OTF
    post-pads the PSF array (down or to the right) with zeros to match
    dimensions specified in OUTSIZE, then circularly shifts the values of
    the PSF array up (or to the left) until the central pixel reaches (1,1)
    position.

    Parameters
    ----------
    psf : `numpy.ndarray`
        PSF array
    shape : int
        Output shape of the OTF array

    Returns
    -------
    otf : `numpy.ndarray`
        OTF array

    Notes
    -----
    Adapted from MATLAB psf2otf function

    """
    if np.all(psf == 0):
        return np.zeros_like(psf)

    inshape = psf.shape
    # Pad the PSF to outsize
    if np.all(np.asarray(shape) != inshape):
        # psf = zero_pad(psf, shape, position='corner')
        r_pad = shape[0] - inshape[0]
        b_pad = shape[1] - inshape[1]
        psf = zero_pad(psf, ((0, b_pad), (0, r_pad)), mode='constant', constant_values=0)

    # Circularly shift OTF so that the 'center' of the PSF is
    # [0,0] element of the array
    for axis, axis_size in enumerate(inshape):
        psf = np.roll(psf, -int(axis_size / 2), axis=axis)

    # Compute the OTF
    otf = np.fft.fft2(psf)

    # Estimate the rough number of operations involved in the FFT
    # and discard the PSF imaginary part if within roundoff error
    # roundoff error  = machine epsilon = sys.float_info.epsilon
    # or np.finfo().eps
    n_ops = np.sum(psf.size * np.log2(psf.shape))
    otf = np.real_if_close(otf, tol=n_ops)

    return otf

def psf2otf_torch(psf, outSize):
    psfSize = torch.tensor(psf.shape)
    outSize = torch.tensor(outSize[-2:])
    padSize = outSize - psfSize
    
    # pad left and right side with zeros
    psf=torch.nn.functional.pad(psf,(0, padSize[1], 0, padSize[0]), 'constant')

    # roll the kernel
    # link: https://www.mathworks.com/help/images/ref/psf2otf.html
    for i in range(len(psfSize)):
        psf = torch.roll(psf, -int(psfSize[i] / 2), i)
        
    # to fft space
    otf = torch.fft.fft2(psf)
    
    nElem = torch.prod(psfSize)
    nOps = 0
    for k in range(len(psfSize)):
        nffts = nElem / psfSize[k]
        nOps = nOps + psfSize[k] * torch.log2(psfSize[k]) * nffts
        
    if torch.max(torch.abs(torch.imag(otf))) / torch.max(torch.abs(otf)) <= nOps * torch.finfo(torch.float32).eps:
        otf = torch.real(otf)
    return otf