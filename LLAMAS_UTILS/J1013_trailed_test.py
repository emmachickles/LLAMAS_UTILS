from llamas_pyjamas.config import BASE_DIR, OUTPUT_DIR
from LLAMAS_UTILS.pyjamas_utils import plot_whitelight, extract_fiber, extract_aper
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import EarthLocation, SkyCoord
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.stats import binned_statistic
import pickle
import pkg_resources
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
# plt.ion()

filepaths = ['/Users/emma/projects/llamas-data/ATLASJ1013/LLAMAS_2024-11-30T08_22_09.466_mef.fits',
             '/Users/emma/projects/llamas-data/ATLASJ1013/LLAMAS_2024-11-30T08_24_06.175_mef.fits',
             '/Users/emma/projects/llamas-data/ATLASJ1013/LLAMAS_2024-11-30T08_25_54.028_mef.fits',
             '/Users/emma/projects/llamas-data/ATLASJ1013/LLAMAS_2024-11-30T08_27_44.893_mef.fits',
             '/Users/emma/projects/llamas-data/ATLASJ1013/LLAMAS_2024-11-30T08_29_35.964_mef.fits',
             '/Users/emma/projects/llamas-data/ATLASJ1013/LLAMAS_2024-11-30T08_31_21.158_mef.fits',
             '/Users/emma/projects/llamas-data/ATLASJ1013/LLAMAS_2024-11-30T08_33_09.110_mef.fits',
             '/Users/emma/projects/llamas-data/ATLASJ1013/LLAMAS_2024-11-30T08_35_03.104_mef.fits',
             '/Users/emma/projects/llamas-data/ATLASJ1013/LLAMAS_2024-11-30T08_36_50.070_mef.fits',
             '/Users/emma/projects/llamas-data/ATLASJ1013/LLAMAS_2024-11-30T08_38_36.523_mef.fits']
# aper = [[23.72, 23.65], [23.72, 23.65], [23.72, 23.65]]
aper = [[23.5, 23.382687], [23.5, 23.382687], [23.5, 23.382687]]
objname = 'J1013'

out_dir = '/Users/emma/Desktop/work/250527/Trailed/'
out_dir += objname+'_'

package_path = pkg_resources.resource_filename('llamas_pyjamas', '')
LUT = package_path+'/Image/LLAMAS_FiberMap_revA.dat'

BJD = []
lines = [4686, 5411, 5568, 5883, 6292, 6554]
wave_line = [[]]*len(lines)
flux_line = [[]]*len(lines)

waves_coadd = []
spectra_coadd = []

for ifile, filepath in enumerate(filepaths):

    ext = filepath.split('/')[-1][:-8]
    extract_pickle = OUTPUT_DIR+'/'+ext+'extract.pkl'
    white_fits = OUTPUT_DIR+'/'+ext+'mef_whitelight.fits'    

    with open(extract_pickle, 'rb') as f: 
        exobj = pickle.load(f)

    # -- get spectra ----------------------------------------------------------
    # waves, spectra = extract_aper(exobj, LUT, aper, out_dir+ext) 
    waves, spectra = extract_fiber(exobj, LUT, aper)
    waves_coadd.append(waves)
    spectra_coadd.append(spectra)

    for iline, line in enumerate(lines):
        inds = np.nonzero( (waves[1] > line-45) * (waves[1] < line+45) )

        wave_line[iline].append( waves[1][inds] )
        flux_line[iline].append( spectra[1][inds] )

    timestamp = ext[7:-1]
    timestamp = timestamp.replace('_', ':')
    t = Time(timestamp, format='isot', scale='utc')
    t = t.tdb
    Observatory=EarthLocation.of_site('Las Campanas Observatory')
    c = SkyCoord(153.42697,-45.28243, unit="deg")
    delta=t.light_travel_time(c,kind='barycentric',location=Observatory)
    t=t+delta   
    BJD.append(t.mjd)

    fig, ax = plt.subplots(nrows=3, figsize=(12, 8))
    ax[0].plot(waves[2], spectra[2], c='b')
    ax[1].plot(waves[1], spectra[1], c='g')
    ax[2].plot(waves[0], spectra[0], c='r')
    ax[2].set_xlabel('Wavelength (Å)') 
    ax[1].set_ylabel('Counts')
    fig.savefig(out_dir+ext+'_counts.png', dpi=300)
    plt.close()

    fig, ax = plt.subplots(ncols=len(lines), figsize=(12, 6))
    for iline, line in enumerate(lines):
        ax[iline].plot(wave_line[iline][ifile], flux_line[iline][ifile], c='g')
        ax[iline].set_xlabel('Wavelength (Å)')
        ax[iline].set_ylabel('Counts')

    fig.savefig(out_dir+ext+'_lines.png', dpi=300)
    plt.close(fig)

    import pdb
    pdb.set_trace()

BJD = np.array(BJD)
wave_line = np.array(wave_line)
flux_line = np.array(flux_line)

period = 0.0059443686582844385
t0 = 57404.55139335168

phi = ((BJD - t0) / period) % 1

for iline, line in enumerate(lines):

    wave_template = wave_line[iline][0]
    flux_interp = []
    for i, waves in enumerate(wave_line[iline]):
        flux = flux_line[iline]
        flux = interp1d(waves, flux, kind='linear', fill_value=np.nan, bounds_error=False)(wave_template)
        flux_interp.append(flux)
    c = 2.99792458e5
    rv = ((wave_template - 4686) / 4686) * c
    flux = np.array(flux_interp)

    phi_extended = np.concatenate([phi - 1, phi])
    flux_extended = np.concatenate([flux, flux], axis=0)  # shape: (20, 157)

    plt.figure(figsize=(6, 3))
    plt.imshow(flux_extended.T, aspect='auto', cmap='inferno',
            extent=[phi_extended.min(), phi_extended.max(), rv.min(), rv.max()],
            origin='lower', interpolation='none')

    plt.colorbar(label='Flux')
    plt.xlabel('Orbital Phase')
    plt.ylabel('Radial Velocity (km/s)')
    plt.title('He II 4686 Å Trailed Spectrum')
    plt.xticks(ticks=np.linspace(-1, 1, 9))
    plt.tight_layout()
    plt.savefig(out_dir+ext+f'_{line}.png', dpi=300)
    plt.show()
