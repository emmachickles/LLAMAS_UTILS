gu_filepath = "/Users/emma/projects/llamas-data/Arc/LLAMAS_2024-11-29T23_07_53.063_mef.fits"
gu_extract_pickle = "/Users/emma/projects/llamas-pyjamas/llamas_pyjamas/output/LLAMAS_2024-11-29T23_07_53.063_extract.pkl"

r_filepath = "/Users/emma/projects/llamas-data/Arc/LLAMAS_2024-12-03T23_02_27.119_mef.fits"
r_extract_pickle = "/Users/emma/projects/llamas-pyjamas/llamas_pyjamas/output/LLAMAS_2024-12-03T23_02_27.119_extract.pkl"


ThArpath = "/Users/emma/projects/LLAMAS_UTILS/LLAMAS_UTILS/ThAr_lines.dat"


# -- extract arc --------------------------------------------------------------
# # Note: before running GUI_extract, must
# # $ conda activate llamas
# # $ samp_hub &
# # $ ds9 &
# from llamas_pyjamas.GUI.guiExtract import GUI_extract
# GUI_extract(gu_filepath)
# GUI_extract(r_filepath)

import pickle
with open(gu_extract_pickle, 'rb') as f:
    gu_exobj = pickle.load(f)
with open(r_extract_pickle, 'rb') as f:
    r_exobj = pickle.load(f)    

red_fiber = r_exobj['extractions'][0].counts[10]
green_fiber = gu_exobj['extractions'][1].counts[10]
blue_fiber = gu_exobj['extractions'][2].counts[10]


# -- load ThAr library --------------------------------------------------------

import numpy as np
data = np.genfromtxt(ThArpath, delimiter='|', dtype=None, autostrip=True,
                     usecols=(1,2,3,4,5,6))
line_wave = data[:,1][1:].astype('float')
line_amp = data[:,4][1:].astype('float')

# -- initial guess ------------------------------------------------------------

bands = [[6900, 9800], [4750, 6900], [3500, 4750]]
red_wave = np.linspace(bands[0][1], bands[0][0], 2048)       
green_wave = np.linspace(bands[1][1], bands[1][0], 2048)
blue_wave = np.linspace(bands[2][1], bands[2][0], 2048)

# ThAr lines in each channel
red_inds = np.nonzero( (line_wave > bands[0][0]) * (line_wave < bands[0][1]) )[0]
green_inds = np.nonzero( (line_wave > bands[1][0]) * (line_wave < bands[1][1]) )[0]
blue_inds = np.nonzero( (line_wave > bands[2][0]) * (line_wave < bands[2][1]) )[0]


# Synthetic spectra
bin_size = np.abs(np.diff(red_wave)[0])
bins = np.arange(bands[0][0]-bin_size/2, bands[0][1]+bin_size/2, bin_size)
red_synth, _ =np.histogram(line_wave[red_inds], bins=bins,
                            weights=line_amp[red_inds])
red_synth = red_synth[::-1]

bin_size = np.abs(np.diff(green_wave)[0])
bins = np.arange(bands[1][0]-bin_size/2, bands[1][1]+bin_size/2, bin_size)
green_synth, _ =np.histogram(line_wave[green_inds], bins=bins,
                            weights=line_amp[green_inds])
green_synth = green_synth[::-1]

bin_size = np.abs(np.diff(blue_wave)[0])
bins = np.arange(bands[2][0]-bin_size/2, bands[2][1]+bin_size/2, bin_size)
blue_synth, _ =np.histogram(line_wave[blue_inds], bins=bins,
                            weights=line_amp[blue_inds])
blue_synth = blue_synth[::-1]


from pypeit.core.wavecal.wvutils import xcorr_shift_stretch

xcorr_shift_stretch(blue_synth, blue_fiber, debug=True)


# red_synth = np.interp(red_wave, line_wave[red_inds], line_amp[red_inds])
# green_synth = np.interp(green_wave, line_wave[green_inds], line_amp[green_inds])
# blue_synth = np.interp(blue_wave, line_wave[blue_inds], line_amp[blue_inds])

# red_synth *= np.nanmax(red_fiber) / np.max(red_synth)
# green_synth *= np.nanmax(green_fiber) / np.max(green_synth)
# blue_synth *= np.nanmax(blue_fiber) / np.max(blue_synth)

# import matplotlib.pyplot as plt
# plt.ion()

# fig, ax = plt.subplots(figsize=(20,12), nrows=2, sharex=True)
# sorted_inds = np.argsort(line_amp[red_inds])[::-1]
# for i in range(20):
#     ax[0].axvline(line_wave[red_inds[sorted_inds[i]]], c='k', lw=1)
# ax[0].plot(red_wave, red_fiber, '-r')
# ax[1].plot(line_wave[red_inds], line_amp[red_inds], '-k', lw=1, alpha=0.2)
# ax[1].plot(red_wave, red_synth, '-k')
# ax[1].set_xlabel('Approx Wavelength')
# ax[0].set_ylabel('Counts')
# ax[1].set_ylabel('Amplitude')
# plt.subplots_adjust(hspace=0)

# fig, ax = plt.subplots(figsize=(20,12), nrows=2, sharex=True)
# sorted_inds = np.argsort(line_amp[green_inds])[::-1]
# for i in range(20):
#     ax[0].axvline(line_wave[green_inds[sorted_inds[i]]], c='k', lw=1)
# ax[0].plot(green_wave, green_fiber, '-g')
# ax[1].plot(line_wave[green_inds], line_amp[green_inds], '-k', lw=1, alpha=0.2)
# ax[1].plot(green_wave, green_synth, '-k')
# ax[1].set_xlabel('Approx Wavelength')
# ax[0].set_ylabel('Counts')
# ax[1].set_ylabel('Amplitude')
# plt.subplots_adjust(hspace=0)

# fig, ax = plt.subplots(figsize=(20,12), nrows=2, sharex=True)
# sorted_inds = np.argsort(line_amp[blue_inds])[::-1]
# for i in range(20):
#     ax[0].axvline(line_wave[blue_inds[sorted_inds[i]]], c='k', lw=1)
# ax[0].plot(blue_wave, blue_fiber, '-b')
# ax[1].plot(line_wave[blue_inds], line_amp[blue_inds], '-k', lw=1, alpha=0.2)
# ax[1].plot(blue_wave, blue_synth, '-k')
# ax[1].set_xlabel('Approx Wavelength')
# ax[0].set_ylabel('Counts')
# ax[1].set_ylabel('Amplitude')
# plt.subplots_adjust(hspace=0)

# -- determine shift and stretch ----------------------------------------------

# from scipy.signal import correlate, correlation_lags

# # Initial guess
# cc = correlate(green_fiber, green_synth, mode="full")
# lags = correlation_lags(len(green_fiber), len(green_synth), mode="full")

# best_shift = lags[np.argmax(cc)]
# calibrated_green_wave = np.roll(green_wave, -best_shift)

# plt.figure(figsize=(20,8))
# plt.plot(calibrated_green_wave, green_synth, '-k', alpha=0.8)
# plt.plot(green_wave, green_fiber, '-g', alpha=0.8)


# cc = correlate(blue_fiber, reference_spectrum, mode="full")



