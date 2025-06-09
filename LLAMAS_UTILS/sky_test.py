from llamas_pyjamas.config import OUTPUT_DIR
from LLAMAS_UTILS.pyjamas_utils import extract_fiber_by_pos, extract_sky, load_arc_pkl, load_LUT
import matplotlib.pyplot as plt
import pkg_resources, pickle
from scipy.interpolate import interp1d
import numpy as np
import sys, os

plt.rcParams['font.family'] = 'serif'

filepaths = [
    '/Users/emma/work/llamas-data/ATLASJ1013/LLAMAS_2024-11-30T08_22_09.466_mef.fits',
    '/Users/emma/work/llamas-data/ATLASJ1013/LLAMAS_2024-11-30T08_24_06.175_mef.fits',
    '/Users/emma/work/llamas-data/ATLASJ1013/LLAMAS_2024-11-30T08_25_54.028_mef.fits',
    '/Users/emma/work/llamas-data/ATLASJ1013/LLAMAS_2024-11-30T08_27_44.893_mef.fits',
    '/Users/emma/work/llamas-data/ATLASJ1013/LLAMAS_2024-11-30T08_29_35.964_mef.fits',
    '/Users/emma/work/llamas-data/ATLASJ1013/LLAMAS_2024-11-30T08_31_21.158_mef.fits',
    '/Users/emma/work/llamas-data/ATLASJ1013/LLAMAS_2024-11-30T08_33_09.110_mef.fits',
    '/Users/emma/work/llamas-data/ATLASJ1013/LLAMAS_2024-11-30T08_35_03.104_mef.fits',
    '/Users/emma/work/llamas-data/ATLASJ1013/LLAMAS_2024-11-30T08_36_50.070_mef.fits',
    '/Users/emma/work/llamas-data/ATLASJ1013/LLAMAS_2024-11-30T08_38_36.523_mef.fits'
]

out_dir = '/Users/emma/Desktop/plots/250604/coadd/'
os.makedirs(out_dir, exist_ok=True)

x, y = [23.5, 23.382687] # target position

# ext = filepath.split('/')[-1][:-8]
# extract_pickle = OUTPUT_DIR+'/'+ext+'extract.pkl'
# white_fits = OUTPUT_DIR+'/'+ext+'mef_whitelight.fits'

fibermap = load_LUT()
arc = load_arc_pkl()

# from LLAMAS_UTILS.pyjamas_utils import plot_whitelight
# from astropy.io import fits
# whitelight = fits.open(white_fits)
# plot_whitelight(whitelight, x, y)
# plt.savefig(out_dir+ext+'_whitelight.png', dpi=300)

# with open(extract_pickle, 'rb') as f: 
#     exobj = pickle.load(f)

# from LLAMAS_UTILS.pyjamas_utils import plot_local_fiber_intensity
# plot_local_fiber_intensity(exobj, fibermap, arc, x, y, out_dir)


# from LLAMAS_UTILS.pyjamas_utils import plot_fiber_intensity_map
# plot_fiber_intensity_map(exobj, fibermap, arc, x, y, out_dir)

# spec = extract_fiber_by_pos(exobj, fibermap, arc, x, y)

# from LLAMAS_UTILS.pyjamas_utils import extract_aper, sky_subtraction

# spec = extract_aper(exobj, fibermap, arc, x, y)
# spec_sky = extract_sky(exobj, fibermap, arc, x, y, spec)
# spec_skysub = sky_subtraction(spec, spec_sky)

# from LLAMAS_UTILS.pyjamas_utils import plot_aperture_fibers
# plot_aperture_fibers(exobj, fibermap, arc, x, y, spec, out_dir+ext+'sci_')
# plot_aperture_fibers(exobj, fibermap, arc, x, y, spec_sky, out_dir+ext+'sky_')
# plot_aperture_fibers(exobj, fibermap, arc, x, y, spec_skysub, out_dir+ext+'skysub_')



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

from LLAMAS_UTILS.pyjamas_utils import extract_all_spectra_and_bjd, plot_skysub_spectra_by_color, plot_line_window_from_skysub

skysub_spectra, bjds = extract_all_spectra_and_bjd(
    filepaths, fibermap, arc, x, y
)

skysub_spectra[0]['r']['mask'] = (skysub_spectra[0]['r']['wave'] < 6957.48) | (skysub_spectra[0]['r']['wave'] > 6966.09)
skysub_spectra[6]['r']['mask'] = ((skysub_spectra[6]['r']['wave'] < 8037.3) | (skysub_spectra[6]['r']['wave'] > 8046.6)) & \
                                    ((skysub_spectra[6]['r']['wave'] < 8684.76) | (skysub_spectra[6]['r']['wave'] > 8693.25))
skysub_spectra[7]['r']['mask'] = (skysub_spectra[0]['r']['wave'] < 8529.82) | (skysub_spectra[0]['r']['wave'] > 8540)
skysub_spectra[8]['r']['mask'] = (skysub_spectra[0]['r']['wave'] < 7380) | (skysub_spectra[0]['r']['wave'] > 7386)
skysub_spectra[7]['g']['mask'] = ((skysub_spectra[0]['g']['wave'] < 4991) | (skysub_spectra[0]['g']['wave'] > 4999)) & \
                                    ((skysub_spectra[0]['g']['wave'] < 5575) | (skysub_spectra[0]['g']['wave'] > 5584))                 
skysub_spectra[0]['b']['mask'] = (skysub_spectra[0]['b']['wave'] < 3871) | (skysub_spectra[0]['b']['wave'] > 3875)
skysub_spectra[3]['b']['mask'] = ((skysub_spectra[0]['b']['wave'] < 3881) | (skysub_spectra[0]['b']['wave'] > 3887)) & \
                                    ((skysub_spectra[0]['b']['wave'] < 4503) | (skysub_spectra[0]['b']['wave'] > 4505))                 
skysub_spectra[6]['b']['mask'] = (skysub_spectra[0]['b']['wave'] < 3879) | (skysub_spectra[0]['b']['wave'] > 3884)

# plot_skysub_spectra_by_color(skysub_spectra, bjds, out_dir)

from LLAMAS_UTILS.pyjamas_utils import plot_trailed_spectrum_from_skysub
period = 0.0059443686582844385
t0 = 57404.55139335168
plot_trailed_spectrum_from_skysub(
    skysub_spectra, bjds, color='b', line_center=4471,
    period=period, t0=t0,
    out_path=out_dir+'HeII_4471_trailed.png',
    label='He II 4471 Å'
)

plot_line_window_from_skysub(
    skysub_spectra, bjds, color='b', line_center=4471,
    out_path=out_dir+'HeII_4686_window.png',
    label='He II 4686 Å',
    t0=t0, period=period
)

plot_trailed_spectrum_from_skysub(
    skysub_spectra, bjds, color='g', line_center=4686,
    period=period, t0=t0, 
    out_path=out_dir+'HeII_4686_trailed.png',
    label='He II 4686 Å'
)

plot_line_window_from_skysub(
    skysub_spectra, bjds, color='g', line_center=4686,
    out_path=out_dir+'HeII_4686_window.png',
    label='He II 4686 Å',
    t0=t0, period=period
)

plot_trailed_spectrum_from_skysub(
    skysub_spectra, bjds, color='g', line_center=5411,
    period=period, t0=t0, 
    out_path=out_dir+'HeII_5411_trailed.png',
    label='He II 5411 Å'
)

plot_line_window_from_skysub(
    skysub_spectra, bjds, color='g', line_center=4686,
    out_path=out_dir+'HeII_5411_window.png',
    label='He II 5411 Å',
    t0=t0, period=period
)

plot_trailed_spectrum_from_skysub(
    skysub_spectra, bjds, color='g', line_center=6560,
    period=period, t0=t0, 
    out_path=out_dir+'HeII_6560_trailed.png',
    label='He II 6560 Å'
)

plot_line_window_from_skysub(
    skysub_spectra, bjds, color='g', line_center=6560,
    out_path=out_dir+'HeII_6560_window.png',
    label='He II 6560 Å',
    t0=t0, period=period
)


from LLAMAS_UTILS.pyjamas_utils import coadd_all_colors, plot_coadd_spectra_by_color

coadd_spec = coadd_all_colors(skysub_spectra)
# plot_coadd_spectra_by_color(coadd_spec, out_dir)


def plot_coadd_spectra_all_channels(coadd_results, lines, line_names, out_path=None, label=None):
    """
    Plot the coadded spectra for all three color channels on a single axes, binning each spectrum.
    Before plotting, fit a 3rd order polynomial to each channel and divide the flux by the fitted continuum.

    Parameters
    ----------
    coadd_results : dict
        Dictionary with keys 'r', 'g', 'b', each containing a tuple (wave, coadd_flux).
    out_path : str or None
        If provided, save the plot to this path.
    label : str or None
        Optional label for the plot title.
    """
    import matplotlib.pyplot as plt
    from scipy.signal import savgol_filter
    import numpy as np

    color_map = {'r': 'red', 'g': 'green', 'b': 'blue'}
    plt.figure(figsize=(14, 5))  # Wide figure for paper
    wave_all = []
    flux_all = []
    for color in ['r', 'g', 'b']:
        wave, coadd_flux = coadd_results[color]

        # Remove overlapping regions
        if color == 'b':
            mask = wave < 4657
            wave = wave[mask]
            coadd_flux = coadd_flux[mask]
        elif color == 'g':
            mask = wave < 6875
            wave = wave[mask]
            coadd_flux = coadd_flux[mask]
        elif color == 'r':
            mask = wave > 6875
            wave = wave[mask]
            coadd_flux = coadd_flux[mask]

        # Fit a 3rd order polynomial to the binned flux (continuum)
        continuum = np.copy(coadd_flux)
        if color == 'g':
            emission = (wave < 4716) | ( (wave > 5389) & (wave < 5454) ) | ( (wave > 6527) & (wave < 6598) )
            continuum[emission] = np.nanmedian(coadd_flux)
        continuum = savgol_filter(continuum, window_length=501, polyorder=3) 
        median_cont = np.nanmedian(continuum)
        rel_flux = (coadd_flux - continuum + median_cont) / median_cont
        
        if color == 'g':
            window_length = 11
        if color == 'r':
            window_length = 25
        if color == 'b':
            window_length = 25
        smoothed_flux = savgol_filter(rel_flux, window_length=window_length, polyorder=2)

        # shift by estimated radial velocity
        wave -= 4693 - 4685.7

        wave_all.extend(wave)
        flux_all.extend(smoothed_flux)

    wave_all = np.array(wave_all)
    flux_all = np.array(flux_all)
    sort_idx = np.argsort(wave_all)
    flux_all = flux_all[sort_idx]
    wave_all = wave_all[sort_idx]

    plt.plot(wave_all, flux_all, '-k', lw=1.2)

    # Draw vertical lines and labels for each line, placing label just above the local maximum in a window
    for line, name in zip(lines, line_names):
        # Find the closest wavelength index
        idx = np.abs(wave_all - line).argmin()
        # Define a window of +/-10 Å around the line center
        window_mask = (wave_all >= line - 10) & (wave_all <= line + 10)
        if np.any(window_mask):
            local_max = np.max(flux_all[window_mask])
        else:
            local_max = flux_all[idx]
        label_y = local_max + 0.08 * (np.max(flux_all) - np.min(flux_all))  # offset above local max
        line_y0 = local_max + 0.04 * (np.max(flux_all) - np.min(flux_all))
        line_y1 = label_y - 0.01 * (np.max(flux_all) - np.min(flux_all))  # end just below label

        plt.vlines(line, line_y0, line_y1, color='gray', linestyle='-', lw=1.5)
        plt.text(line, label_y, name, color='black', fontsize=13, ha='center', va='bottom', rotation=0)

    # Add extra space above the spectrum for labels
    y_max = np.max(flux_all)
    plt.ylim(np.min(flux_all), y_max + 0.2 * (y_max - np.min(flux_all)))
    plt.xlabel('Wavelength (Å)', fontsize=16)
    plt.ylabel('Relative Intensity', fontsize=16)
    plt.tight_layout() 
    plt.xlim(np.min(wave_all), np.max(wave_all))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()

lines      = [4035,
              4187.79,
              4387.93,
              4471.48,
              4685.7,
              4859,
              4921.93,
              5412,
              5876,
              6559.7,
              8236.77,
              9224.57,
              9229.02]
line_names = ['N II',
              'He I',
              'He I',
              'He I',
              'He II',
              'He II',
              'He I',
              'He II',
              'He I',
              'He II',
              'He II',
              'He II',
              'He I']
# plot_coadd_spectra_all_channels(coadd_spec, lines, line_names, out_path=out_dir+'coadd_all_channels.png')