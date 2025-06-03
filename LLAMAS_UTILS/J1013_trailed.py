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

out_dir = '/Users/emma/Desktop/work/250424/'
out_dir += objname+'_'

package_path = pkg_resources.resource_filename('llamas_pyjamas', '')
LUT = package_path+'/Image/LLAMAS_FiberMap_revA.dat'

BJD = []
wave_HeII = []
flux_HeII = []
wave_HeI = []
flux_HeI = []

waves_coadd = []
spectra_coadd = []

for filepath in filepaths:

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

    inds_HeII = np.nonzero( (waves[1] > 4686-45) * (waves[1] < 4686+45) )
    inds_HeI = np.nonzero( (waves[1] > 5411-50) * (waves[1] < 5411+50) )

    wave_HeII.append( waves[1][inds_HeII] )
    flux_HeII.append( spectra[1][inds_HeII] )
    wave_HeI.append( waves[1][inds_HeI] )
    flux_HeI.append( spectra[1][inds_HeI] )

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

    fig, ax = plt.subplots(ncols=2, figsize=(12, 6))
    ax[0].plot(waves[1][inds_HeII], spectra[1][inds_HeII], c='g')
    ax[0].set_xlabel('Wavelength (Å)')
    ax[0].set_ylabel('Counts')
    
    ax[1].plot(waves[1][inds_HeI], spectra[1][inds_HeI], c='g')
    ax[1].set_xlabel('Wavelength (Å)')
    ax[1].set_ylabel('Counts')    
    fig.savefig(out_dir+ext+'_He.png', dpi=300)
    plt.close(fig)

BJD = np.array(BJD)
wave_HeII = np.array(wave_HeII)
flux_HeII = np.array(flux_HeII)
wave_HeI = np.array(wave_HeI)
flux_HeI = np.array(flux_HeI)

period = 0.0059443686582844385
t0 = 57404.55139335168

phi = ((BJD - t0) / period) % 1

wave_template = wave_HeII[0]
flux_HeII_interp = []
for i, waves in enumerate(wave_HeII):
    flux = flux_HeII[i]
    flux = interp1d(waves, flux, kind='linear', fill_value=np.nan, bounds_error=False)(wave_template)
    flux_HeII_interp.append(flux)
c = 2.99792458e5
rv_HeII = ((wave_template - 4686) / 4686) * c
flux_HeII = np.array(flux_HeII_interp)

phi_extended = np.concatenate([phi - 1, phi])
flux_HeII_extended = np.concatenate([flux_HeII, flux_HeII], axis=0)  # shape: (20, 157)

plt.figure(figsize=(6, 3))
plt.imshow(flux_HeII_extended.T, aspect='auto', cmap='inferno',
           extent=[phi_extended.min(), phi_extended.max(), rv_HeII.min(), rv_HeII.max()],
           origin='lower', interpolation='none')

plt.colorbar(label='Flux')
plt.xlabel('Orbital Phase')
plt.ylabel('Radial Velocity (km/s)')
plt.title('He II 4686 Å Trailed Spectrum')
plt.xticks(ticks=np.linspace(-1, 1, 9))
plt.tight_layout()
plt.savefig(out_dir+ext+'_He_4686_trailed.png', dpi=300)
plt.show()

wave_template = wave_HeI[0]
flux_HeI_interp = []
for i, waves in enumerate(wave_HeI):
    flux = flux_HeI[i]
    flux = interp1d(waves, flux, kind='linear', fill_value=np.nan, bounds_error=False)(wave_template)
    flux_HeI_interp.append(flux)
c = 2.99792458e5
rv_HeI = ((wave_template - 4686) / 4686) * c
flux_HeI = np.array(flux_HeI_interp)

flux_HeI_extended = np.concatenate([flux_HeI, flux_HeI], axis=0)  

plt.figure(figsize=(6, 3))
plt.imshow(flux_HeI_extended.T, aspect='auto', cmap='inferno',
           extent=[phi_extended.min(), phi_extended.max(), rv_HeII.min(), rv_HeII.max()],
           origin='lower', interpolation='none')

plt.colorbar(label='Flux')
plt.xlabel('Orbital Phase')
plt.ylabel('Radial Velocity (km/s)')
plt.title('He II 5411 Å Trailed Spectrum')
plt.xticks(ticks=np.linspace(-1, 1, 9))
plt.tight_layout()
plt.savefig(out_dir+ext+'_He_5411_trailed.png', dpi=300)
plt.show()