from llamas_pyjamas.config import OUTPUT_DIR
from LLAMAS_UTILS.pyjamas_utils import extract_fiber_by_pos, extract_sky, load_arc_pkl, load_LUT
import matplotlib.pyplot as plt
import pkg_resources, pickle
from scipy.interpolate import interp1d
import numpy as np
import sys, os

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
out_dir = '/Users/emma/Desktop/plots/250604/squared_weight/'
out_dir += objname+'_'
os.makedirs(out_dir, exist_ok=True)

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

# from LLAMAS_UTILS.pyjamas_utils import plot_local_fiber_intensity
# plot_local_fiber_intensity(exobj, fibermap, arc, x, y, out_dir)


# from LLAMAS_UTILS.pyjamas_utils import plot_fiber_intensity_map
# plot_fiber_intensity_map(exobj, fibermap, arc, x, y, out_dir)

# spec = extract_fiber_by_pos(exobj, fibermap, arc, x, y)

from LLAMAS_UTILS.pyjamas_utils import extract_aper, sky_subtraction

spec = extract_aper(exobj, fibermap, arc, x, y)
spec_sky = extract_sky(exobj, fibermap, arc, x, y, spec)
spec_skysub = sky_subtraction(spec, spec_sky)

from LLAMAS_UTILS.pyjamas_utils import plot_aperture_fibers
plot_aperture_fibers(exobj, fibermap, arc, x, y, spec, out_dir+ext+'sci_')
plot_aperture_fibers(exobj, fibermap, arc, x, y, spec_sky, out_dir+ext+'sky_')
plot_aperture_fibers(exobj, fibermap, arc, x, y, spec_skysub, out_dir+ext+'skysub_')



# from LLAMAS_UTILS.pyjamas_utils import identify_emission_lines_with_pypeit, plot_emission_line_fits
# idx = np.nonzero( (spec_sky['g']['bench'] == '4A') * (spec_sky['g']['fiber']==202))[0][0]
# lines = identify_emission_lines_with_pypeit(spec_sky['g']['wave'], spec_sky['g']['all_spectra'][idx])
# plot_emission_line_fits(spec_sky['g']['wave'], spec_sky['g']['all_spectra'][idx], lines,
#                         out_dir+ext+f'bench{spec_sky['g']['bench'][idx]}_fiber{spec_sky['g']['fiber'][idx]}_')

# idx = np.nonzero( (spec_sky['g']['bench'] == '4B') * (spec_sky['g']['fiber']==257))[0][0]
# lines = identify_emission_lines_with_pypeit(spec_sky['g']['wave'], spec_sky['g']['all_spectra'][idx])
# plot_emission_line_fits(spec_sky['g']['wave'], spec_sky['g']['all_spectra'][idx], lines,
#                         out_dir+ext+f'bench{spec_sky['g']['bench'][idx]}_fiber{spec_sky['g']['fiber'][idx]}_')


# from LLAMAS_UTILS.pyjamas_utils import refine_wavelength_with_telluric, plot_wavelength_refinement_fit
# new_waves, fit_info = refine_wavelength_with_telluric(spec_sky['g']['wave'], spec_sky['g']['all_spectra'][idx])
# plot_wavelength_refinement_fit(fit_info, out_dir+ext)

# lines = identify_emission_lines_with_pypeit(new_waves, spec_sky['g']['all_spectra'][idx])
# plot_emission_line_fits(new_waves, spec_sky['g']['all_spectra'][idx], lines,
#                         out_dir+ext+f'bench{spec_sky['g']['bench'][idx]}_fiber{spec_sky['g']['fiber'][idx]}_telluric_')


# from LLAMAS_UTILS.pyjamas_utils import debug_extract_sky_fit, analyze_emission_lines_across_fibers
# analyze_emission_lines_across_fibers(spec_sky, out_dir=out_dir+ext)
# debug_extract_sky_fit(spec_sky, exobj, fibermap, arc, x, y, out_dir+ext)

# from LLAMAS_UTILS.pyjamas_utils import plot_interp_vs_raw_fiber
# plot_interp_vs_raw_fiber(spec_sky, exobj, arc, 'g', '4B', 257, out_dir+ext)



# plt.ion()
# for i, c in enumerate(['r', 'g', 'b']):
#     fig, ax = plt.subplots(nrows=3, figsize=(12, 4), sharex=True)
#     ax[0].plot(spec[c]['wave'], spec[c]['spectrum'], label='Science Aperture', color=c)
#     ax[1].plot(spec[c]['wave'], spec_sky[c]['spectrum'], label='Sky Aperture', color=c)
#     ax[2].plot(spec[c]['wave'], spec[c]['spectrum'] - spec_sky[c]['spectrum'],
#             label='Sky Subtracted', color=c)
#     ax[2].set_xlabel('Wavelength (Å)') 
#     ax[1].set_ylabel('Counts')
#     ax[0].legend()
#     ax[1].legend()
#     ax[2].legend()
#     fig.savefig(out_dir+ext+f'skysub_{c}.png', dpi=300)