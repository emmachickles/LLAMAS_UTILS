# Experiment using xcorr 

b_filepath = "/Users/emma/projects/llamas-data/Arc/LLAMAS_2024-12-02T09_47_39.341_mef.fits"
b_extract_pickle = "/Users/emma/projects/llamas-pyjamas/llamas_pyjamas/output/LLAMAS_2024-12-02T09_47_39.341_extract.pkl"

ThArpath = "/Users/emma/projects/LLAMAS_UTILS/LLAMAS_UTILS/ThAr_lines.dat"
out_dir = "/Users/emma/Desktop/work/250317/"

# -- load libraries -----------------------------------------------------------

import numpy as np
import pickle 
from scipy.ndimage import gaussian_filter
from pypeit.core.wavecal.wvutils import xcorr_shift_stretch, shift_and_stretch, get_xcorr_arc, xcorr_shift, zerolag_shift_stretch
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

plt.ion()

# -- arc exposure -------------------------------------------------------------

with open(b_extract_pickle, 'rb') as f:
    b_exobj = pickle.load(f)  
    
fiber = 150
blue_spec = b_exobj['extractions'][20].counts[fiber]

# -- initial guess ------------------------------------------------------------

bands = [[6900, 9800], [4600, 7200], [3425, 5100]] 
blue_wave = np.linspace(bands[2][1], bands[2][0], 2048)

# -- load ThAr library --------------------------------------------------------

data = np.genfromtxt(ThArpath, delimiter='|', dtype=None, autostrip=True,
                     usecols=(1,2,3,4,5,6))
ThAr_wave = data[:,1][1:].astype('float')
ThAr_amp = data[:,4][1:].astype('float')

# ThAr lines in each channel
blue_inds = np.nonzero( (ThAr_wave > bands[2][0]) * (ThAr_wave < bands[2][1]) )[0]

# Synthetic spectra
R = 2000
blue_syn_wave = ThAr_wave[blue_inds]
sigma = np.mean(blue_wave) / R
blue_syn = gaussian_filter(ThAr_amp[blue_inds], sigma)
blue_syn = np.interp(blue_wave, blue_syn_wave, blue_syn)
blue_syn *= np.nanmax(blue_spec) / np.max(blue_syn)

# -- wave solution ------------------------------------------------------------

inspec1 = blue_syn
inspec2 = blue_spec

bad_inds = np.nonzero(np.isnan(inspec2))
good_inds = np.nonzero(~np.isnan(inspec2))
inspec2[bad_inds] = np.interp( np.arange(inspec2.size)[bad_inds],
                               np.arange(inspec2.size)[good_inds], inspec2[good_inds])

inds = np.nonzero( np.arange(inspec2.size) > 1700 )
inspec2[inds] *= 2
inds = np.nonzero( np.arange(inspec2.size) < 1330 )
inspec2[inds] *= 0.5

percent_ceil=50.0
sigdetect=15.0
sig_ceil=10.0
fwhm=8.0

y1 = get_xcorr_arc(inspec1, percent_ceil=percent_ceil,
                      sigdetect=sigdetect, sig_ceil=sig_ceil, fwhm=fwhm)
inds = np.nonzero( np.arange(y1.size) < 390 )
y1[inds] = np.median(y1)

percent_ceil=50.0
sigdetect=20.0
sig_ceil=10.0
fwhm=8.0
y2 = get_xcorr_arc(inspec2, percent_ceil=percent_ceil,
                   sigdetect=sigdetect, sig_ceil=sig_ceil, fwhm=fwhm)

fig, ax = plt.subplots(nrows=2, figsize=(20,8), sharex=True)
ax[0].set_ylabel('inspec1 / y1')
ax[1].set_ylabel('inspec2 / y2')
ax[0].plot(inspec1)
ax[1].plot(inspec2)
ax[0].plot(y1)
ax[1].plot(y2)
ax[0].invert_xaxis()
fig.subplots_adjust(hspace=0)

shift_cc, corr_cc = xcorr_shift(y1, y2, do_xcorr_arc=False, 
                                sigdetect=sigdetect, fwhm=fwhm, debug=True)
y2_corr = shift_and_stretch(y2, shift_cc, 1., 0.)

fig, ax = plt.subplots(nrows=2, figsize=(20,8), sharex=True)
ax[0].set_ylabel('y1')
ax[1].set_ylabel('y2 / y2_corr')
ax[0].plot(y1)
ax[1].plot(y2)
ax[1].plot(y2_corr)
ax[0].invert_xaxis()
fig.subplots_adjust(hspace=0)

nspec = inspec1.size
shift_mnmx = (-0.2, 0.2)
stretch_mnmx = (0.85, 1.15)
toler = 1e-5

lag_range = (shift_cc + nspec * shift_mnmx[0], shift_cc + nspec * shift_mnmx[1])
bounds = [lag_range, stretch_mnmx, (-1.0e-6, 1.0e-6)]
x0_guess = np.array([shift_cc, 1.0, 0.0])

result = differential_evolution(zerolag_shift_stretch, args=(y1,y2),
                                x0=x0_guess, tol=toler, bounds=bounds,
                                disp=False, polish=True)

y2_corr = shift_and_stretch(y2, result.x[0], result.x[1], result.x[2])
fig, ax = plt.subplots(nrows=2, figsize=(20,8), sharex=True)
ax[0].set_ylabel('y1')
ax[1].set_ylabel('y2_corr')
ax[0].plot(y1)
ax[1].plot(y2_corr)
ax[0].invert_xaxis()
fig.subplots_adjust(hspace=0)
