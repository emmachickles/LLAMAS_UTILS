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
from LLAMAS_UTILS.pyjamas_utils import wavecal
from scipy.ndimage import gaussian_filter
from matplotlib.widgets import Slider
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

bands = [[6900, 9800], [4700, 6920], [3500, 5200]]
red_wave = np.linspace(bands[0][1], bands[0][0], 2048)       
green_wave = np.linspace(bands[1][1], bands[1][0], 2048)
blue_wave = np.linspace(bands[2][1], bands[2][0], 2048)

waves = wavecal()
red_wave = waves[0]
green_wave = waves[1]
blue_wave = waves[2]
bands = [[np.min(red_wave), np.max(red_wave)],
         [np.min(green_wave), np.max(green_wave)],
         [np.min(blue_wave), np.max(blue_wave)]]

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

# Synthetic spectra
red_synth_wave = line_wave[red_inds]
sigma = np.mean(red_wave) / 2500
red_synth = gaussian_filter(line_amp[red_inds], sigma)
red_synth = np.interp(red_wave, red_synth_wave, red_synth)
red_synth *= np.nanmax(red_fiber) / np.max(red_synth)

green_synth_wave = line_wave[green_inds]
sigma = np.mean(green_wave) / 2500
green_synth = gaussian_filter(line_amp[green_inds], sigma)
green_synth = np.interp(green_wave, green_synth_wave, green_synth)
green_synth *= np.nanmax(green_fiber) / np.max(green_synth)

blue_synth_wave = line_wave[blue_inds]
sigma = np.mean(green_wave) / 2500
blue_synth = gaussian_filter(line_amp[blue_inds], sigma)
blue_synth = np.interp(blue_wave, blue_synth_wave, blue_synth)
blue_synth *= np.nanmax(blue_fiber) / np.max(blue_synth)


for i, c in enumerate(['r', 'g', 'b']):
    if c == 'r':
        inds = red_inds
        wave = red_wave
        fiber = red_fiber
        synth = red_synth
    elif c == 'g':
        inds = green_inds
        wave = green_wave
        fiber = green_fiber
        synth = green_synth
    elif c == 'b':
        inds = blue_inds
        wave = blue_wave
        fiber = blue_fiber
        synth = blue_synth
    
    fig, ax = plt.subplots(figsize=(20,8), nrows=2, sharex=True)
    ax[0].plot(wave, fiber, '-'+c)
    ax[1].plot(wave, synth, '-k')
    ax[1].set_xlabel('Approx Wavelength')
    ax[0].set_ylabel('Counts')
    ax[1].set_ylabel('Amplitude')
    plt.subplots_adjust(hspace=0)


# -- wave calibration ---------------------------------------------------------

from pypeit.core.wavecal.wvutils import xcorr_shift_stretch, shift_and_stretch

# Sliders
c='b'
if c == 'r':
    wave = red_wave
    fiber = red_fiber
    synth = red_synth
elif c == 'g':
    wave = green_wave
    fiber = green_fiber
    synth = green_synth
elif c == 'b':
    wave = blue_wave
    fiber = blue_fiber
    synth = blue_synth    


inds = np.nonzero(~np.isnan(fiber))
wave, fiber, synth = wave[inds], fiber[inds], synth[inds]

fig, ax = plt.subplots(figsize=(20,8))
plt.subplots_adjust(bottom=0.3)  # Space for sliders\
ax.plot(wave, synth, '-k')
line_shifted, = ax.plot(wave, fiber, '-'+c, alpha=0.7)
ax.set_xlabel('Wavelength')
ax.set_ylabel('Counts')

# Add sliders
ax_shift = plt.axes([0.2, 0.15, 0.65, 0.03])
ax_stretch = plt.axes([0.2, 0.1, 0.65, 0.03])
ax_stretch2 = plt.axes([0.2, 0.05, 0.65, 0.03])    

slider_shift = Slider(ax_shift, 'Shift', -500, 500, valinit=0)
slider_stretch = Slider(ax_stretch, 'Stretch', 0.2, 1.8, valinit=1)
slider_stretch2 = Slider(ax_stretch2, 'Stretch2', -0.001, 0.001, valinit=0)
    
# Update function for sliders
def update(val):
    shift = slider_shift.val
    stretch = slider_stretch.val
    stretch2 = slider_stretch2.val
    new_spec = shift_and_stretch(fiber, shift, stretch, stretch2)
    
    print('MSE: '+str(np.round(np.mean((new_spec-synth)**2),1)))
    
    line_shifted.set_ydata(new_spec)
    # plt.draw()
    fig.canvas.draw_idle() 

# Connect sliders to update function
slider_shift.on_changed(update)
slider_stretch.on_changed(update)
slider_stretch2.on_changed(update)    

plt.draw()


# for i, c in enumerate(['g', 'b']): # enumerate(['r', 'g', 'b']):
#     if c == 'r':
#         wave = red_wave
#         fiber = red_fiber
#         synth = red_synth
#     elif c == 'g':
#         wave = green_wave
#         fiber = green_fiber
#         synth = green_synth
#     elif c == 'b':
#         wave = blue_wave
#         fiber = blue_fiber
#         synth = blue_synth    
    

#     result_out, shift_out, stretch_out, stretch2_out, corr_out, shift_cc, corr_cc = \
#         xcorr_shift_stretch(synth, fiber, toler=1e-8,
#                             shift_mnmx=(-0.01,0.01), stretch_mnmx=(0.85, 1.15))
#     print(shift_out)
#     print(stretch_out)
#     print(stretch2_out)
    
#     fiber_corr = shift_and_stretch(fiber, shift_out, stretch_out, stretch2_out)
    
#     fig, ax = plt.subplots(figsize=(20,12), nrows=3, sharex=True)
#     ax[0].plot(wave, fiber, '--k', label='Raw arc')
#     ax[0].plot(wave, fiber_corr, '-'+c, label='Shifted and stretched arc')
#     ax[1].plot(wave, synth, '-.k', label='ThAr library')
#     ax[2].plot(wave, fiber_corr, '-'+c, label='Shifted and stretched arc')
#     ax[2].plot(wave, synth, '-.k', label='ThAr library')
#     ax[2].set_xlabel('Wavelength')
#     ax[0].set_ylabel('Counts')
#     ax[1].set_ylabel('Amplitude')
#     ax[0].legend()
#     ax[1].legend()
#     ax[2].legend()
#     plt.subplots_adjust(hspace=0)
    
#     fiber =  fiber_corr
        
  