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

out_dir = "/Users/emma/Desktop/work/250414/"

import numpy as np
from scipy.signal import correlate, correlation_lags
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from LLAMAS_UTILS.pyjamas_utils import wavecal
from scipy.ndimage import gaussian_filter
from matplotlib.widgets import Slider
from scipy.optimize import curve_fit
from scipy.signal import correlate, correlation_lags
from scipy.ndimage import shift
from scipy.interpolate import interp1d
plt.ion()

# bench_list = np.array(['1A', '1B', '2A', '2B', '3A', '3B', '4A', '4B'])
bench_list = np.array(['4A', '4B'])


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

fiber = 150

red_spec = r_exobj['extractions'][18].counts[fiber]
print(r_exobj['metadata'][18]['channel'])
green_spec = g_exobj['extractions'][19].counts[fiber]
print(g_exobj['metadata'][19]['channel'])
blue_spec = b_exobj['extractions'][20].counts[fiber]
print(b_exobj['metadata'][20]['channel'])




# -- initial guess ------------------------------------------------------------

bands = [[6900, 9800], [4600, 7200], [3425, 5100]] 
red_wave = np.linspace(bands[0][1], bands[0][0], 2048)       
green_wave = np.linspace(bands[1][1], bands[1][0], 2048)
blue_wave = np.linspace(bands[2][1], bands[2][0], 2048)

# waves = wavecal()
# red_wave = waves[0]
# green_wave = waves[1]
# blue_wave = waves[2]
# bands = [[np.min(red_wave), np.max(red_wave)],
#          [np.min(green_wave), np.max(green_wave)],
#          [np.min(blue_wave), np.max(blue_wave)]]

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
R = 2000
red_synth_wave = line_wave[red_inds]
sigma = np.mean(red_wave) / R
red_synth = gaussian_filter(line_amp[red_inds], sigma)
red_synth = np.interp(red_wave, red_synth_wave, red_synth)
red_synth *= np.nanmax(red_spec) / np.max(red_synth)

green_synth_wave = line_wave[green_inds]
sigma = np.mean(green_wave) / R
green_synth = gaussian_filter(line_amp[green_inds], sigma)
green_synth = np.interp(green_wave, green_synth_wave, green_synth)
green_synth *= np.nanmax(green_spec) / np.max(green_synth)

blue_synth_wave = line_wave[blue_inds]
sigma = np.mean(green_wave) / R
blue_synth = gaussian_filter(line_amp[blue_inds], sigma)
blue_synth = np.interp(blue_wave, blue_synth_wave, blue_synth)
blue_synth *= np.nanmax(blue_spec) / np.max(blue_synth)


# for i, c in enumerate(['r', 'g', 'b']):
#     if c == 'r':
#         inds = red_inds
#         wave = red_wave
#         spec = red_spec
#         synth = red_synth
#     elif c == 'g':
#         inds = green_inds
#         wave = green_wave
#         spec = green_spec
#         synth = green_synth
#     elif c == 'b':
#         inds = blue_inds
#         wave = blue_wave
#         spec = blue_spec
#         synth = blue_synth
    
#     fig, ax = plt.subplots(figsize=(20,8), nrows=2, sharex=True)
#     ax[0].plot(wave, spec, '-'+c)
#     ax[1].plot(wave, synth, '-k')
#     ax[1].set_xlabel('Approx Wavelength')
#     ax[0].set_ylabel('Counts')
#     ax[1].set_ylabel('Amplitude')
#     plt.subplots_adjust(hspace=0)


# -- wave calibration ---------------------------------------------------------

from pypeit.core.wavecal.wvutils import xcorr_shift_stretch, shift_and_stretch, arc_lines_from_spec

# # Sliders
# c='b'
# if c == 'r':
#     wave = red_wave
#     spec = red_spec
#     synth = red_synth
# elif c == 'g':
#     wave = green_wave
#     spec = green_spec
#     synth = green_synth
# elif c == 'b':
#     wave = blue_wave
#     spec = blue_spec
#     synth = blue_synth    


# good_inds = np.nonzero(~np.isnan(spec))
# wave, spec, synth = wave[good_inds], spec[good_inds], synth[good_inds]

# fig, ax = plt.subplots(figsize=(20,8))
# plt.subplots_adjust(bottom=0.3)  # Space for sliders\
# ax.plot(wave, synth, '-k')
# line_shifted, = ax.plot(wave, spec, '-'+c, alpha=0.7)
# ax.set_xlabel('Wavelength')
# ax.set_ylabel('Counts')

# # Add sliders
# ax_shift = plt.axes([0.2, 0.15, 0.65, 0.03])
# ax_stretch = plt.axes([0.2, 0.1, 0.65, 0.03])
# ax_stretch2 = plt.axes([0.2, 0.05, 0.65, 0.03])    

# slider_shift = Slider(ax_shift, 'Shift', -500, 500, valinit=0)
# slider_stretch = Slider(ax_stretch, 'Stretch', 0.2, 1.8, valinit=1)
# slider_stretch2 = Slider(ax_stretch2, 'Stretch2', -0.001, 0.001, valinit=0)
    
# # Update function for sliders
# def update(val):
#     shift = slider_shift.val
#     stretch = slider_stretch.val
#     stretch2 = slider_stretch2.val
#     new_spec = shift_and_stretch(spec, shift, stretch, stretch2)
    
#     print('MSE: '+str(np.round(np.mean((new_spec-synth)**2),1)))
    
#     line_shifted.set_ydata(new_spec)
#     # plt.draw()
#     fig.canvas.draw_idle() 

# # Connect sliders to update function
# slider_shift.on_changed(update)
# slider_stretch.on_changed(update)
# slider_stretch2.on_changed(update)    

# plt.draw()


for c in ['g', 'b']: # enumerate(['r', 'g', 'b']):
    if c == 'r':
        inds = red_inds
        wave = red_wave
        spec = red_spec
        synth = red_synth
    elif c == 'g':
        inds = green_inds
        wave = green_wave
        spec = green_spec
        synth = green_synth
        
        sigdetect = 20
        syn_sigdetect = 20
    elif c == 'b':
        inds = blue_inds
        wave = blue_wave
        spec = blue_spec
        synth = blue_synth    
        
        sigdetect_lo = 12
        sigdetect_up = 46
        syn_sigdetect = 16
    
        
    if c == 'b':
        tcent_lo, ecent, cut_tcent, icut, arc_cont_sub = \
            arc_lines_from_spec(spec, sigdetect=sigdetect_lo)   
        tcent_up, ecent, cut_tcent, icut, arc_cont_sub = \
            arc_lines_from_spec(spec, sigdetect=sigdetect_up)     
        tcent = np.append(tcent_lo[tcent_lo > 1700],
                          tcent_up[tcent_up < 1700])
    else:
        tcent, ecent, cut_tcent, icut, arc_cont_sub = \
            arc_lines_from_spec(spec, sigdetect=sigdetect)
    # Returns
    # tcent        :  centroids of the line detections
    # ecent        :  variance on tcent
    # cut_tsent    :  tcent[ nsig > sigdetect ]
    # icut         :  indices
    # arc_cont_sub :  continuum subtracted arc
    
    wave_cent = np.interp(tcent, np.arange(len(spec)), wave)
    flux_cent = np.interp(tcent, np.arange(len(spec)), spec)
    
    syn_tcent, syn_ecent, syn_cut_tcent, syn_icut, syn_arc_cont_sub = \
        arc_lines_from_spec(synth, sigdetect=syn_sigdetect)    
    syn_wave_cent = np.interp(syn_tcent, np.arange(len(spec)), wave)
    syn_flux_cent = np.interp(syn_tcent, np.arange(len(spec)), synth)
    
    fig, ax = plt.subplots(figsize=(20,8), nrows=2, sharex=True)
    for i in range(len(tcent)):
        ax[0].axvline(wave_cent[i], c='k', alpha=0.5)    
        ax[0].text(wave_cent[i]+2, flux_cent[i], str(np.round(tcent[i],1)),
                   rotation='vertical', ha='left', va='bottom')
    ax[0].plot(wave, spec, '-'+c)   
    # ax[1].plot(line_wave[inds], line_amp[inds], '-k', alpha=0.5)
    for i in range(len(syn_tcent)):
        ax[1].axvline(syn_wave_cent[i], c='r', alpha=0.5)        
        ax[1].text(syn_wave_cent[i]+2, syn_flux_cent[i], str(np.round(syn_wave_cent[i],1)),
                   rotation='vertical', ha='left', va='bottom')
    ax[1].plot(wave, synth, '-k')
    ax[1].set_xlabel('Approx Wavelength')
    ax[0].set_ylabel('Counts')
    ax[1].set_ylabel('Amplitude')
    plt.subplots_adjust(hspace=0)
    plt.savefig(out_dir+c+'_'+str(fiber)+'_peaks.png', dpi=300)
    plt.close()    
    
    if c == 'g':
        pix_cent = [1568.7, 1590.6, 1722.3, 2006.7,
                    1034.9, 1013.7, 984.3, 968.2, 819.8, 809.5, 576.1, 512.5, 466.7, 461.8,
                    1525.7, 1500.4, 1440.9, 1375.6, 1342.0, 1326.5, 1314.9, 1303.2, 1194.3, 1161.0, 1125.4]
        arc_cent = [5259.8, 5232.6, 5066.8, 4705.4,
                    5916.0, 5940.8, 5975.8, 5996.2, 6171.4, 6183.7, 6458.8, 6533.1, 6585.8, 6592.5,
                    5313.9, 5345.1, 5419.0, 5501.1, 5540.9, 5559.0, 5574.2, 5588.9, 5721.8, 5761.7, 5805.8]
    
    if c == 'b':
        pix_cent = [1839.9, 1854.7, # 1839.9, 1847.7, 1854.7,
            1678.8, 1752.8, 1774.2,
            1574.1, 1541.8, 1527.9, 1484.0, 1388.3, 1297.0,
            1174.2, 1020.5,
            584]
        arc_cent = [3589.8, 3577.7, # 3589.8, 3577.7, 3568.0,
            3722.5, 3660.8, 3643.4,
            3804.2, 3829.2, 3840.2, 3875.3, 3951.1, 4020.3,
            4114.2, 4212.5,
            4494.3]
        
    
    
    pix_cent = np.array(pix_cent)
    arc_cent = np.array(arc_cent)
    
    
    sorted_inds = np.argsort(pix_cent)
    pix_cent, arc_cent = pix_cent[sorted_inds], arc_cent[sorted_inds]
    
    cmap = cm.get_cmap('viridis', len(pix_cent))
    
    fig, ax = plt.subplots(figsize=(20,8), nrows=2)
    for i in range(len(pix_cent)):
        ax[0].axvline(pix_cent[i], c=cmap(i))
        ax[1].axvline(arc_cent[i], c=cmap(i))
    ax[0].plot(np.arange(len(spec)), spec, '-'+c)   
    ax[0].invert_xaxis()
    ax[1].plot(wave, synth, '-k')
    ax[0].set_xlabel('Pixel number')
    ax[1].set_xlabel('Approx Wavelength')
    ax[0].set_ylabel('Counts')
    ax[1].set_ylabel('Amplitude')       
    plt.savefig(out_dir+c+'_'+str(fiber)+'_lines.png', dpi=300)
    plt.close()
    
    def func(x, a, b, c):
        return a*x**2 + b*x + c
    
    popt, pcov = curve_fit(func, pix_cent, arc_cent)
    # print(popt)
    # print(pcov)
    
    
    new_wave = func(np.arange(len(spec)), popt[0], popt[1], popt[2])
    
    np.savetxt('wavecal_solutions/template_'+c+'_4A_'+str(fiber)+'.txt', new_wave)    
    
    if c == 'r':
        red_wave_sol = new_wave
    elif c == 'g':
        green_wave_sol = new_wave
    elif c == 'b':
        blue_wave_sol = new_wave
    
    fig, ax = plt.subplots(figsize=(8,6))
    ax.set_title(c)
    ax.plot(pix_cent, arc_cent, '.')    
    ax.plot(np.arange(len(spec)), new_wave)    
    ax.set_xlabel('Pixel')
    ax.set_ylabel('Wavelength')
    plt.savefig(out_dir+c+'_'+str(fiber)+'_soln.png', dpi=300)    
    plt.close()

    
    fig, ax = plt.subplots(figsize=(20,8), nrows=2, sharex=True)
    ax[0].plot(new_wave, spec, '-'+c)
    # ax[1].plot(line_wave[inds], line_amp[inds], '-k')
    ax[1].plot(wave, synth, '-k')
    ax[1].set_xlabel('Wavelength')
    ax[0].set_ylabel('Counts')
    ax[1].set_ylabel('Amplitude')
    plt.subplots_adjust(hspace=0)            
    plt.savefig(out_dir+c+'_'+str(fiber)+'_calib.png', dpi=300)    
    plt.close()    
    
    # good_inds = np.nonzero(~np.isnan(spec))
    # wave, spec, synth = wave[good_inds], spec[good_inds], synth[good_inds]

    # result_out, shift_out, stretch_out, stretch2_out, corr_out, shift_cc, corr_cc = \
    #     xcorr_shift_stretch(synth, spec, sigdetect=sigdetect)
    # print(shift_out)
    # print(stretch_out)
    # print(stretch2_out)
    
    # spec_corr = shift_and_stretch(spec, shift_out, stretch_out, stretch2_out)
    
    # fig, ax = plt.subplots(figsize=(20,12), nrows=3, sharex=True)
    # ax[0].plot(wave, spec, '--k', label='Raw arc')
    # ax[0].plot(wave, spec_corr, '-'+c, label='Shifted and stretched arc')
    # ax[1].plot(wave, synth, '-.k', label='ThAr library')
    # ax[2].plot(wave, spec_corr, '-'+c, label='Shifted and stretched arc')
    # ax[2].plot(wave, synth, '-.k', label='ThAr library')
    # ax[2].set_xlabel('Wavelength')
    # ax[0].set_ylabel('Counts')
    # ax[1].set_ylabel('Amplitude')
    # ax[0].legend()
    # ax[1].legend()
    # ax[2].legend()
    # plt.subplots_adjust(hspace=0)
    
# -- get solutions ------------------------------------------------------------

for bench in bench_list:
    
    red_hduidx = np.nonzero(bench_list == bench)[0][0]*3
    green_hduidx = np.nonzero(bench_list == bench)[0][0]*3 + 1
    blue_hduidx = np.nonzero(bench_list == bench)[0][0]*3 + 2
    
    for fiber in range(298):
        
        print(bench)
        print(fiber)
        
        # fiber = 100
        # red_spec1 = r_exobj['extractions'][18].counts[fiber]
        # green_spec1 = g_exobj['extractions'][19].counts[fiber]
        # blue_spec1 = b_exobj['extractions'][20].counts[fiber]
        
        # if len(r_exobj['extractions'][red_hduidx].counts) >= fiber-1:
        #     red_spec1 = r_exobj['extractions'][red_hduidx].counts[fiber]
        if len(g_exobj['extractions'][green_hduidx].counts) >= fiber-1:
            green_spec1 = g_exobj['extractions'][green_hduidx].counts[fiber]
        if len(b_exobj['extractions'][blue_hduidx].counts) >= fiber-1:
            blue_spec1 = b_exobj['extractions'][blue_hduidx].counts[fiber]    
        
        for c in ['g', 'b']:
            # if c == 'r':
            #     inds = red_inds
            #     wave = red_wave
            #     spec = red_spec
            #     spec1 = red_spec1
            #     synth = red_synth
            #     wave_sol = red_wave_sol
            if c == 'g':
                inds = green_inds
                wave = green_wave
                spec = green_spec
                spec1 = green_spec1
                synth = green_synth
                wave_sol = green_wave_sol
            elif c == 'b':
                inds = blue_inds
                wave = blue_wave
                spec = blue_spec
                spec1 = blue_spec1
                synth = blue_synth
                wave_sol = blue_wave_sol
                
            nan_inds = np.nonzero(np.isnan(spec))[0]
            num_inds = np.nonzero(~np.isnan(spec))[0]
            
            spec[nan_inds] = np.interp(nan_inds, num_inds, spec[num_inds])
        
            nan_inds = np.nonzero(np.isnan(spec1))[0]
            num_inds = np.nonzero(~np.isnan(spec1))[0]
            
            spec1[nan_inds] = np.interp(nan_inds, num_inds, spec1[num_inds])
        
            # # Compute the cross-correlation
            # corr = correlate(spec1, spec, mode='full')
            
            # # Compute the lags
            # lags = correlation_lags(len(spec1), len(spec), mode='full')
        
            # # Find the shift corresponding to the peak
            # p = np.polyfit(lags, corr, 5)
            # lag_interp = np.linspace(np.min(lags), np.max(lags), 20000)
            # corr_interp = np.polyval(p, lag_interp)
            # shift_out = lag_interp[np.argmax(corr_interp)]
            
            # plt.figure()
            # plt.plot(lags, corr, '-k')
            # plt.plot(lag_interp, corr_interp, lw=1)
            # plt.xlabel('Lags')
            # plt.ylabel('Correlation')
            
            # spec_corr = shift(spec1, shift=-shift_out, mode='nearest')
            # new_wave_sol = shift(wave_sol, shift=shift_out, mode='nearest')
            
              
            percent_ceil=80.0
            sigdetect=20.0
            sig_ceil=10.0
            fwhm=8.0
            
            result_out, shift_out, stretch_out, stretch2_out, corr_out, shift_cc, corr_cc = \
                xcorr_shift_stretch(spec, spec1, percent_ceil=percent_ceil, sig_ceil=sig_ceil,
                                    debug=False, sigdetect=sigdetect, fwhm=fwhm) 
                
            spec1_corr = shift_and_stretch(spec1, shift_out, stretch_out, stretch2_out)
            # new_wave_sol = shift_and_stretch(wave_sol, shift_out, stretch_out, stretch2_out)
            
            nspec = spec1.size
            old_x = np.arange(nspec)
            new_x = np.arange(nspec)**2*stretch2_out + np.arange(nspec)*stretch_out + shift_out
            # new_x = np.arange(nspec)
            # old_x = np.arange(nspec)**2*stretch2_out + np.arange(nspec)*stretch_out + shift_out
            new_wave_sol = interp1d(old_x, wave_sol, kind='quadratic',
                                    fill_value='extrapolate')(new_x)
                  
            
            np.savetxt('wavecal_solutions/'+c+'_'+bench+'_'+str(fiber)+'.txt', new_wave_sol)  
            
            
            fig, ax = plt.subplots(nrows=2, figsize=(20,8), sharex=True)
            ax[0].set_ylabel('spec')
            ax[1].set_ylabel('spec1 / spec1_corr')
            ax[0].plot(spec)
            ax[1].plot(spec1)
            ax[1].plot(spec1_corr)
            ax[0].invert_xaxis()
            fig.subplots_adjust(hspace=0)
            fig.savefig(out_dir+c+'_'+bench+'_'+str(fiber)+'_spec.png', dpi=300)
            plt.close()
            
            # fig, ax = plt.subplots(nrows=2, figsize=(20,8), sharex=True)
            # ax[0].set_ylabel('synth')
            # ax[1].set_ylabel('spec1')
            # ax[0].plot(wave, synth, '-k')
            # ax[1].plot(new_wave_sol, spec1, '-'+c)
            # fig.subplots_adjust(hspace=0)    
        
            # fig, ax = plt.subplots(nrows=2, figsize=(20,8), sharex=True)
            # ax[0].set_ylabel('spec')
            # ax[1].set_ylabel('spec1')
            # ax[0].plot(wave_sol, spec, '-'+c)
            # ax[1].plot(new_wave_sol, spec1, '-'+c)
            # fig.subplots_adjust(hspace=0)      
        

        
