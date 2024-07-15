import asdf
import numpy as np
from astropy.stats import SigmaClip
from astropy.convolution import Gaussian1DKernel

def smooth(spec, fwhm):
    gaus = Gaussian1DKernel(fwhm/2.355)
    return np.convolve(spec, gaus)


fname = input('Put the name of the spectrum: ')
thred_user = float(input('Put the threshold value: '))
fwhm = float(input('Put the FWHM value: '))

file = asdf.open(fname)
wave = np.array(np.asarray(file['wave']),dtype='<f8')
flux = np.array(np.asarray(file['flux']),dtype='<f8')
#err = np.array(np.asarray(file['err']),dtype='<f8')

chan_width = wave[1]-wave[0]

absorption = flux - 1 

smooth_abs = smooth(absorption, fwhm)

clip = SigmaClip(sigma=2, maxiters=5)
clipped_smooth_abs = clip(smooth_abs)

rms_smooth_abs = np.std(clipped_smooth_abs)

threshold = thred_user * rms_smooth_abs


