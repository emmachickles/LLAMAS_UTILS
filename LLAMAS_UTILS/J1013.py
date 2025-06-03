from llamas_pyjamas.config import BASE_DIR, OUTPUT_DIR
from LLAMAS_UTILS.pyjamas_utils import plot_whitelight, extract_fiber, extract_aper, plot_fiber_intensity_map
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

out_dir = '/Users/emma/Desktop/work/250527/'
out_dir += objname+'_'

package_path = pkg_resources.resource_filename('llamas_pyjamas', '')
LUT = package_path+'/Image/LLAMAS_FiberMap_revA.dat'

BJD = []
wave0_HeII = []

waves_coadd = []
spectra_coadd = []

for filepath in filepaths[:3]:

    ext = filepath.split('/')[-1][:-8]
    extract_pickle = OUTPUT_DIR+'/'+ext+'extract.pkl'
    white_fits = OUTPUT_DIR+'/'+ext+'mef_whitelight.fits'    

    with open(extract_pickle, 'rb') as f: 
        exobj = pickle.load(f)

    # -- plot whitelight ------------------------------------------------------
    # whitelight = fits.open(white_fits)
    # plot_whitelight(whitelight, aper)
    # plt.savefig(out_dir+ext+'_whitelight.png', dpi=300)
    plot_fiber_intensity_map(exobj, LUT, aper, out_dir+ext)

    # -- get spectra ----------------------------------------------------------
    # waves, spectra = extract_aper(exobj, LUT, aper, out_dir+ext) 
    waves, spectra = extract_fiber(exobj, LUT, aper)
    waves_coadd.append(waves)
    spectra_coadd.append(spectra)

    inds_HeII = np.nonzero( (waves[1] > 4686-150) * (waves[1] < 4686+150) )
    inds_HeI = np.nonzero( (waves[1] > 5411-150) * (waves[1] < 5411+150) )
    
    def gaussian(x, amp, mu, sigma, offset):
        return amp * np.exp(-(x - mu)**2 / (2 * sigma**2)) + offset
    popt, pcov = curve_fit(gaussian, waves[1][inds_HeII], spectra[1][inds_HeII],
                           p0=[10, 4686, 1, 0],
                           bounds=[(0, 4686-150, 0, -np.inf), (50, 4686+150, np.inf, np.inf)])
    print(popt)
    xmod = np.linspace(waves[1][inds_HeII].min(), waves[1][inds_HeII].max(), 1000)
    ymod = gaussian(xmod, *popt)
    wave0_HeII.append(popt[1])

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
    for line in [5568, 5581, 6292, 6554]:
        ax[1].axvline(line)
    fig.savefig(out_dir+ext+'_counts.png', dpi=300)
    # plt.close()
    plt.show()

    fig, ax = plt.subplots(ncols=2, figsize=(12, 6))
    ax[0].plot(waves[1][inds_HeII], spectra[1][inds_HeII], c='g')
    ax[0].plot(xmod, ymod, c='r')
    ax[0].set_xlabel('Wavelength (Å)')
    ax[0].set_ylabel('Counts')
    
    ax[1].plot(waves[1][inds_HeI], spectra[1][inds_HeI], c='g')
    ax[1].set_xlabel('Wavelength (Å)')
    ax[1].set_ylabel('Counts')    
    fig.savefig(out_dir+ext+'_HeII.png', dpi=300)
    plt.close(fig)
    

BJD = np.array(BJD)
wave0_HeII = np.array(wave0_HeII)

period = 0.0059443686582844385
t0 = 57404.55139335168

phi = ((BJD - t0) / period) % 1

# convert to RV
c = 2.99792458e5
rv = ((wave0_HeII - 4686) / 4686) * c

# fit a sine wave to the data
def sine_wave(x, A, phi0, offset):
    return A * np.sin(2 * np.pi * x + phi0) + offset    
popt, pcov = curve_fit(sine_wave, phi, rv,
                       p0=[500, 0, 4686],
                       bounds=[(0, 0, -np.inf), (np.inf,  1, np.inf)])
print('RV Semi-amplitude: ', popt[0])
print('RV Offset: ', popt[2])
xmod = np.linspace(phi.min(), phi.max(), 1000)
ymod = sine_wave(xmod, *popt)


# -- plot HeII wavelength vs phase -----------------------------------------

plt.figure()
plt.plot(phi, rv, 'o')
plt.plot(xmod, ymod, c='r')
plt.xlabel('Orbital phase')
plt.ylabel('Radial velocity (km/s)')
plt.title('HeII 4686 Å')
plt.savefig(out_dir+objname+'_HeII.png', dpi=300)

plt.figure()
plt.plot(BJD, wave0_HeII, 'o')
plt.xlabel('BJD')
plt.ylabel('Central Wavelength (Å)')
plt.title('HeII 4686 Å')
plt.savefig(out_dir+objname+'_HeII_central.png', dpi=300)

# -- coadded g spectrum -------------------------------------------------
wave_template = waves_coadd[0][1]
flux_g = []
for i, waves in enumerate(waves_coadd):
    flux = spectra_coadd[i][1]
    flux = interp1d(waves[1], flux, kind='linear', fill_value=np.nan, bounds_error=False)(wave_template)
    flux_g.append(flux)
flux_g = np.array(flux_g)
flux_g = np.sum(flux_g, axis=0)

fig, ax = plt.subplots(figsize=(12, 6))
plt.plot(wave_template, flux_g, c='g')
plt.xlabel('Wavelength (Å)')
plt.ylabel('Summed Counts')
plt.ylim([0,100])
plt.savefig(out_dir+objname+'_g_summed.png', dpi=300)

inds = np.nonzero( wave_template<5500)

# binned_flux, _, _ = binned_statistic(wave_template[inds], flux_g[inds], statistic='mean', bins=700)
# binned_wave, _, _ = binned_statistic(wave_template[inds], wave_template[inds], statistic='mean', bins=700)

wave_shifted = wave_template[inds] + (4686- 4678.4)

lines = [4686, 5411]
line_names = ['He II', 'He II']

fig, ax = plt.subplots(figsize=(6, 4))
# plt.plot(binned_wave, binned_flux, c='g')
plt.plot(wave_shifted, flux_g[inds], c='g')
plt.xlabel('Wavelength (Å)')
plt.ylabel('Summed Counts')
ylims = plt.ylim()
for i in range(len(lines)):
    plt.vlines(lines[i], 15, np.max(flux_g[inds]), ls='--', color='gray')
    plt.text(lines[i], 13, line_names[i], ha='center', va='top')
# plt.ylim([-10,60])
plt.ylim(ylims)
plt.tight_layout()
plt.savefig(out_dir+objname+'_g_shifted_coadd.png', dpi=300)
plt.show()

# -- coadded b spectrum -------------------------------------------------
wave_template = waves_coadd[0][2]
flux_b = []
for i, waves in enumerate(waves_coadd):
    flux = spectra_coadd[i][2]
    flux = interp1d(waves[2], flux, kind='linear', fill_value=np.nan, bounds_error=False)(wave_template)
    flux_b.append(flux)
flux_b = np.array(flux_b)
flux_b = np.sum(flux_b, axis=0)

fig, ax = plt.subplots(figsize=(12, 6))
plt.plot(wave_template, flux_b, c='b')
plt.xlabel('Wavelength (Å)')
plt.ylabel('Summed Counts')
plt.ylim([-10,60])
plt.savefig(out_dir+objname+'_b_summed.png', dpi=300)

binned_flux, _, _ = binned_statistic(wave_template, flux_b, statistic='mean', bins=700)
binned_wave, _, _ = binned_statistic(wave_template, wave_template, statistic='mean', bins=700)

fig, ax = plt.subplots(figsize=(12, 6))
plt.plot(binned_wave, binned_flux, c='b')
plt.xlabel('Wavelength (Å)')
plt.ylabel('Summed Counts')
plt.ylim([-10,60])
plt.savefig(out_dir+objname+'_b_binned.png', dpi=300)