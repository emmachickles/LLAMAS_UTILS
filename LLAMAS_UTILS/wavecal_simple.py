# exptime = 20 s
b_filepath = "/Users/emma/projects/llamas-data/Arc/LLAMAS_2024-12-02T09_47_39.341_mef.fits"
b_extract_pickle = "/Users/emma/projects/llamas-pyjamas/llamas_pyjamas/output/LLAMAS_2024-12-02T09_47_39.341_extract.pkl"

# exptime = 0.2 s
g_filepath = "/Users/emma/projects/llamas-data/Arc/LLAMAS_2024-11-29T23_07_53.063_mef.fits"
g_extract_pickle = "/Users/emma/projects/llamas-pyjamas/llamas_pyjamas/output/LLAMAS_2024-11-29T23_07_53.063_extract.pkl"

# exptime = 0.1 s 
# r_filepath = "/Users/emma/projects/llamas-data/Arc/LLAMAS_2024-12-02T09_37_28.220_mef.fits"
# r_extract_pickle = "/Users/emma/projects/llamas-pyjamas/llamas_pyjamas/output/LLAMAS_2024-12-02T09_37_28.220_extract.pkl"

# exptime = 0.1 s
r_filepath = "/Users/emma/projects/llamas-data/Arc/LLAMAS_2024-12-03T17_02_25.478_mef.fits"
r_extract_pickle = "/Users/emma/projects/llamas-pyjamas/llamas_pyjamas/output/LLAMAS_2024-12-03T17_02_25.478_extract.pkl"


# # exptime = 1 s
# r_filepath = "/Users/emma/projects/llamas-data/Arc/LLAMAS_2024-12-03T17_05_52.592_mef.fits"
# r_extract_pickle = "/Users/emma/projects/llamas-pyjamas/llamas_pyjamas/output/LLAMAS_2024-12-03T17_05_52.592_extract.pkl"

# # exptime = 0.05 s
# r_filepath = "/Users/emma/projects/llamas-data/Arc/LLAMAS_2024-12-03T23_02_27.119_mef.fits"
# r_extract_pickle = "/Users/emma/projects/llamas-pyjamas/llamas_pyjamas/output/LLAMAS_2024-12-03T23_02_27.119_extract.pkl"

ThArpath = "/Users/emma/projects/LLAMAS_UTILS/LLAMAS_UTILS/ThAr_lines.dat"
# ThArpath = "/Users/emma/projects/LLAMAS_UTILS/LLAMAS_UTILS/ThAr_MagE_lines.dat"

out_dir = "/Users/emma/Desktop/work/250219/"

import numpy as np
from scipy.signal import correlate, correlation_lags
import matplotlib.pyplot as plt
plt.ion()



# # -- extract arc --------------------------------------------------------------
# # Note: before running GUI_extract, must
# # $ conda activate llamas
# # $ samp_hub &
# # $ ds9 &
# from llamas_pyjamas.GUI.guiExtract import GUI_extract
# # GUI_extract(b_filepath)
# # GUI_extract(g_filepath)
# GUI_extract(r_filepath)

# # import pdb
# # pdb.set_trace()


import pickle
with open(b_extract_pickle, 'rb') as f:
    b_exobj = pickle.load(f)   
with open(g_extract_pickle, 'rb') as f:
    g_exobj = pickle.load(f)
with open(r_extract_pickle, 'rb') as f:
    r_exobj = pickle.load(f)    

red_fiber = r_exobj['extractions'][18].counts[150]
print(r_exobj['metadata'][18]['channel'])
green_fiber = g_exobj['extractions'][19].counts[150]
print(g_exobj['metadata'][19]['channel'])
blue_fiber = b_exobj['extractions'][20].counts[150]
print(b_exobj['metadata'][20]['channel'])




# -- initial guess ------------------------------------------------------------

bands = [[6900, 9800], [4700, 6920], [3500, 4700]]
red_wave = np.linspace(bands[0][1], bands[0][0], 2048)       
green_wave = np.linspace(bands[1][1], bands[1][0], 2048)
blue_wave = np.linspace(bands[2][1], bands[2][0], 2048)

# -- load ThAr library --------------------------------------------------------




data = np.genfromtxt(ThArpath, delimiter='|', dtype=None, autostrip=True,
                     usecols=(1,2,3,4,5,6))
# line_wave = data[:,0][1:].astype('float')
line_wave = data[:,1][1:].astype('float')
line_amp = data[:,4][1:].astype('float')

# ThAr lines in each channel
red_inds = np.nonzero( (line_wave > bands[0][0]) * (line_wave < bands[0][1]) )[0]
green_inds = np.nonzero( (line_wave > bands[1][0]) * (line_wave < bands[1][1]) )[0]
blue_inds = np.nonzero( (line_wave > bands[2][0]) * (line_wave < bands[2][1]) )[0]

from scipy.ndimage import gaussian_filter
for i, c in enumerate(['r', 'g', 'b']):
    if c == 'r':
        inds = red_inds
        wave = red_wave
        fiber = red_fiber
    elif c == 'g':
        inds = green_inds
        wave = green_wave
        fiber = green_fiber
    elif c == 'b':
        inds = blue_inds
        wave = blue_wave
        fiber = blue_fiber
    
    fig, ax = plt.subplots(figsize=(20,8), nrows=2, sharex=True)
    # sorted_inds = np.argsort(line_amp[inds])[::-1]
    # for i in range(20):
    #     ax[0].axvline(line_wave[inds[sorted_inds[i]]], c='k', lw=1)
    ax[0].plot(wave, fiber, '-'+c)
    
    scaled_lines = line_amp[inds]
    scaled_lines *= np.nanmax(fiber) / np.max(scaled_lines)
    ax[1].plot(line_wave[inds], scaled_lines, '-k', lw=1, alpha=0.5)
    
    sigma = np.mean(wave) / 2000
    scaled_lines = gaussian_filter(line_amp[inds], sigma)
    scaled_lines *= np.nanmax(fiber) / np.max(scaled_lines)
    ax[1].plot(line_wave[inds], scaled_lines, '-', lw=1)
    ax[1].set_xlabel('Approx Wavelength')
    ax[0].set_ylabel('Counts')
    ax[1].set_ylabel('Amplitude')
    plt.subplots_adjust(hspace=0)


# # Synthetic spectra
# bin_size = np.abs(np.diff(red_wave)[0])
# bins = np.linspace(bands[0][0]-bin_size/2, bands[0][1]+bin_size/2, 2049)
# red_synth, _ =np.histogram(line_wave[red_inds], bins=bins,
#                             weights=line_amp[red_inds])
# red_synth = red_synth[::-1]
# red_synth *= np.nanmax(red_fiber) / np.max(red_synth)

# bin_size = np.abs(np.diff(green_wave)[0])
# bins = np.arange(bands[1][0]-bin_size/2, bands[1][1]+bin_size/2, bin_size)
# green_synth, _ =np.histogram(line_wave[green_inds], bins=bins,
#                             weights=line_amp[green_inds])
# green_synth = green_synth[::-1]
# green_synth *= np.nanmax(green_fiber) / np.max(green_synth)

# bin_size = np.abs(np.diff(blue_wave)[0])
# bins = np.arange(bands[2][0]-bin_size/2, bands[2][1]+bin_size/2, bin_size)
# blue_synth, _ =np.histogram(line_wave[blue_inds], bins=bins,
#                             weights=line_amp[blue_inds])
# blue_synth = blue_synth[::-1]
# blue_synth *= np.nanmax(blue_fiber) / np.max(blue_synth)

# # -- wave calibration ------------------------------------------------

# cc = correlate(red_synth, red_fiber, mode="full")
# lags = correlation_lags(2048, 2048, mode='full')
# best_shift = lags[np.argmax(cc)]
# plt.figure()
# plt.plot(np.arange(2048), red_synth, '-k', alpha=0.8)
# plt.plot(np.arange(2048)-best_shift, red_fiber, '-r', alpha=0.6)

# cc = correlate(green_synth, green_fiber, mode="full")
# lags = correlation_lags(2048, 2048, mode='full')
# best_shift = lags[np.argmax(cc)]
# plt.figure()
# plt.plot(np.arange(2048), green_synth, '-k', alpha=0.8)
# plt.plot(np.arange(2048)-best_shift, green_fiber, '-g', alpha=0.6)

# cc = correlate(blue_synth, blue_fiber, mode="full")
# lags = correlation_lags(2048, 2048, mode='full')
# best_shift = lags[np.argmax(cc)]
# plt.figure()
# plt.plot(np.arange(2048), blue_synth, '-k', alpha=0.8)
# plt.plot(np.arange(2048)-best_shift, blue_fiber, '-b', alpha=0.6)


# for i in range(len(r_exobj['extractions'][0].counts)):
#     fig, ax = plt.subplots(figsize=(12,8))
#     plt.plot(red_wave, r_exobj['extractions'][0].counts[i], '-r')
#     plt.savefig(out_dir+'r_exobj_'+str(i)+'.png')
#     plt.close()
    
# for i in range(len(b_exobj['extractions'][2].counts)):
#     fig, ax = plt.subplots(figsize=(12,8))
#     plt.plot(blue_wave, b_exobj['extractions'][2].counts[i], '-b')
#     plt.savefig(out_dir+'b_exobj_blue_'+str(i)+'.png')
#     plt.close()

