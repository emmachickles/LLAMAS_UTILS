from llamas_pyjamas.config import OUTPUT_DIR
from LLAMAS_UTILS.pyjamas_utils import extract_fiber_by_pos, extract_sky, load_arc_pkl, load_LUT
import matplotlib.pyplot as plt
import pkg_resources, pickle
from scipy.interpolate import interp1d
import numpy as np
import sys

# filepath = '/Users/emma/work/llamas-data/ATLASJ1013/LLAMAS_2024-11-30T08_22_09.466_mef.fits'
filepath = '/Users/emma/work/llamas-data/ATLASJ1013/LLAMAS_2024-11-30T08_24_06.175_mef.fits'
# filepath = '/Users/emma/work/llamas-data/ATLASJ1013/LLAMAS_2024-11-30T08_25_54.028_mef.fits'
# filepath = '/Users/emma/work/llamas-data/ATLASJ1013/LLAMAS_2024-11-30T08_27_44.893_mef.fits'
# filepath = '/Users/emma/work/llamas-data/ATLASJ1013/LLAMAS_2024-11-30T08_29_35.964_mef.fits'
# filepath = '/Users/emma/work/llamas-data/ATLASJ1013/LLAMAS_2024-11-30T08_31_21.158_mef.fits'
# filepath = '/Users/emma/work/llamas-data/ATLASJ1013/LLAMAS_2024-11-30T08_33_09.110_mef.fits'
# filepath = '/Users/emma/work/llamas-data/ATLASJ1013/LLAMAS_2024-11-30T08_35_03.104_mef.fits'
# filepath = '/Users/emma/work/llamas-data/ATLASJ1013/LLAMAS_2024-11-30T08_36_50.070_mef.fits'
# filepath = '/Users/emma/work/llamas-data/ATLASJ1013/LLAMAS_2024-11-30T08_38_36.523_mef.fits'


objname = 'J1013'
out_dir = '/Users/emma/Desktop/plots/250602/'
out_dir += objname+'_'

x, y = [23.5, 23.382687] # target position

ext = filepath.split('/')[-1][:-8]
extract_pickle = OUTPUT_DIR+'/'+ext+'extract.pkl'
white_fits = OUTPUT_DIR+'/'+ext+'mef_whitelight.fits'

fibermap = load_LUT()
arc = load_arc_pkl()

# from LLAMAS_UTILS.pyjamas_utils import plot_whitelight
# from astropy.io import fits
# whitelight = fits.open(white_fits)
# plot_whitelight(whitelight, x, y)
# plt.savefig(out_dir+ext+'_whitelight.png', dpi=300)

with open(extract_pickle, 'rb') as f: 
    exobj = pickle.load(f)

from LLAMAS_UTILS.pyjamas_utils import plot_local_fiber_intensity
plot_local_fiber_intensity(exobj, fibermap, arc, x, y, out_dir)


# from LLAMAS_UTILS.pyjamas_utils import plot_fiber_intensity_map
# plot_fiber_intensity_map(exobj, fibermap, arc, x, y, out_dir)

spec = extract_fiber_by_pos(exobj, fibermap, arc, x, y)
spec_sky = extract_sky(exobj, fibermap, arc, x, y, spec)

plt.ion()
for i, c in enumerate(['r', 'g', 'b']):
    fig, ax = plt.subplots(nrows=3, figsize=(12, 4), sharex=True)
    ax[0].plot(spec[c]['wave'], spec[c]['spectrum'], label='Science Aperture', color=c)
    ax[1].plot(spec[c]['wave'], spec_sky[c]['spectrum'], label='Sky Aperture', color=c)
    ax[2].plot(spec[c]['wave'], spec[c]['spectrum'] - spec_sky[c]['spectrum'],
            label='Sky Subtracted', color=c)
    ax[2].set_xlabel('Wavelength (Å)') 
    ax[1].set_ylabel('Counts')
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    fig.savefig(out_dir+ext+c+'_skysub.png', dpi=300)