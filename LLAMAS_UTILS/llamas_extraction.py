# Based on lalmas_pyjamas/Tutorials/llamas_extraction_demo.ipynb

# filepath = '/Users/emma/projects/llamas-test/standard/LLAMAS_2024-11-28T01_22_09.108_mef.fits'
# aper = [[22.51, 22.73], [22.51, 22.73], [22.51, 22.73]]
# objname = 'F110'

# filepath = '/Users/emma/projects/llamas-test/ATLASJ1013/LLAMAS_2024-11-30T08_22_09.466_mef.fits'
# aper = [[23.72, 23.65], [23.72, 23.65], [23.72, 23.65]]
# objname = 'J1013'

filepath = '/Users/emma/projects/llamas-test/ATLASJ1138/LLAMAS_2024-11-28T07_41_00.294_mef.fits'
aper = [[11.27, 19.1], [11.75, 20], [11.27, 20.87]]
objname = 'J1138'


choose_brightest = False


out_dir = '/Users/emma/Desktop/work/250214/'
out_dir +=objname+'_'

# -----------------------------------------------------------------------------

import os
import sys
import numpy as np
import pickle
import pkg_resources

from llamas_pyjamas.config import BASE_DIR, OUTPUT_DIR
sys.path.append(BASE_DIR+'/')

from LLAMAS_UTILS.pyjamas_utils import plot_whitelight, run_extract, \
    extract_aper, extract_fiber, get_fiber, fluxcal, wavecal
from astropy.io import fits

from scipy.stats import binned_statistic
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.ion()


package_path = pkg_resources.resource_filename('llamas_pyjamas', '')
print(f"Package path: {package_path}")

ext = filepath.split('/')[-1][:-8]
LUT = package_path+'/Image/LLAMAS_FiberMap_revA.dat' # look up table
extract_pickle = OUTPUT_DIR+'/'+ext+'extract.pkl'
white_fits = OUTPUT_DIR+'/'+ext+'mef_whitelight.fits'

colors = ['red', 'green', 'blue']

# -- extraction ---------------------------------------------------------------

if not os.path.exists(extract_pickle):
    run_extract(filepath)
    
with open(extract_pickle, 'rb') as f:
    exobj = pickle.load(f)
    
# -- aperture -----------------------------------------------------------------

# Plot white light image
whitelight = fits.open(white_fits)
if choose_brightest:
    aper = get_fiber(whitelight)
plot_whitelight(whitelight, aper)

# Get spectra
spectra = extract_fiber(exobj, LUT, aper)

# Estimate wavelength calibration
waves = wavecal()


# from LLAMAS_UTILS.pyjamas_utils import get_sensfunc

# sensfunc = get_sensfunc()


# Estimate flux calibration
calib_spectra = fluxcal(waves, spectra)

# Save spectrum
for i, color in enumerate(colors):
    data = np.array([waves[i], spectra[i]]).T
    np.savetxt(out_dir+ext+color+'_counts.txt', data)    
    print('Saved '+out_dir+ext+color+'_counts.txt')
    
    data = np.array([waves[i], calib_spectra[i]]).T
    np.savetxt(out_dir+ext+color+'_fluxcal.txt', data)
    print('Saved '+out_dir+ext+color+'_fluxcal.txt')

# Plot science spectrum
fig, ax = plt.subplots(figsize=(20,5))
plt.suptitle('Raw LLAMAS Spectrum of '+objname)

fig_binned, ax_binned = plt.subplots(figsize=(20,5))
plt.suptitle('Flux-Calibrated LLAMAS Spectrum of '+objname)

detector = np.array(['1A', '1B', '2A', '2B', '3A', '3B', '4A', '4B'])

for i, color in enumerate(colors):
    wave = waves[i]
    flux = spectra[i]
    calib_flux = calib_spectra[i]
    
    # Plot counts
    ax.plot(wave, flux, '-', color=color)  
    
    # Plot binned flux-calibrated spectrum    
    binned_w = binned_statistic(wave, wave, bins=256).statistic
    binned_f = binned_statistic(wave, calib_flux, bins=256,
                                statistic='median').statistic   
    if color == 'blue':
        inds = np.nonzero( (binned_w < 4644) * (binned_w > 3571))
        binned_w, binned_f = binned_w[inds], binned_f[inds]
    if color == 'green':
        inds = np.nonzero( binned_w < 6950 )
        binned_w, binned_f = binned_w[inds], binned_f[inds]       
        
    ax_binned.plot(binned_w, binned_f, '-', color=color)
    
ax.set_xlabel('Wavelength (Å)')
ax.set_ylabel('Counts')
fig.tight_layout()
fig.savefig(out_dir+ext+'_counts.png', dpi=300)

ax_binned.set_xlabel('Wavelength (Å)')
ax_binned.set_ylabel(r'Flux (erg cm$^{-2}$ s$^{-1}$ Å$^{-1}$ $\times$ 10$^{16}$)')
fig_binned.tight_layout()
fig_binned.savefig(out_dir+ext+'_fluxcal.png', dpi=300)
