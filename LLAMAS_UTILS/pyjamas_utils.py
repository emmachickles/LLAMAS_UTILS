import numpy as np

aperture_radius = 2.5
aperture_innersky = 3.5
aperture_outersky = 4.5
detector = np.array(['1A', '1B', '2A', '2B', '3A', '3B', '4A', '4B'])

def get_fiber(whitelight):
    aper = []
    
    for i in [5,3,1]:
        
        img = np.fliplr( whitelight[5].data )
        maxcnt = np.nanmax(img)
        xy = np.where(img == maxcnt)
        xy = [xy[1][0]/1.5, xy[0][0]/1.5]
        aper.append(xy)
    
    return aper



def plot_whitelight(whitelight, aper):

    from astropy.io import fits
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig, ax = plt.subplots(figsize=(20,5), ncols=3)
    ax[0].set_title(whitelight[5].header['EXTNAME'])
    ax[0].imshow(np.fliplr(whitelight[5].data), origin='lower', aspect='equal',
                 extent=[0, 80/1.5, 0, 80/1.5])
    ax[1].set_title(whitelight[3].header['EXTNAME'])
    ax[1].imshow(np.fliplr(whitelight[3].data), origin='lower', aspect='equal',
                 extent=[0, 80/1.5, 0, 80/1.5])
    ax[2].set_title(whitelight[1].header['EXTNAME'])
    ax[2].imshow(np.fliplr(whitelight[1].data), origin='lower', aspect='equal',
                 extent=[0, 80/1.5, 0, 80/1.5])
    for i in range(3):        
        ax[i].plot([aper[i][0]], [aper[i][1]], '.m')
        circle = patches.Circle((aper[i][0],aper[i][1]), aperture_radius,
                                ec='m', fc='none')
        ax[i].add_patch(circle)
        circle = patches.Circle((aper[i][0],aper[i][1]), aperture_innersky,
                                ec='k', fc='none')
        ax[i].add_patch(circle)
        circle = patches.Circle((aper[i][0],aper[i][1]), aperture_outersky,
                                ec='k', fc='none')
        ax[i].add_patch(circle)    

        ax[i].set_xlabel('xpos')
        ax[i].set_ylabel('ypos')     


    # fibermap = np.genfromtxt(LUT, delimiter='|', dtype=None, autostrip=True,
    #                          usecols=(1,2,5,6)).astype('str')[1:]
    # bench = fibermap[:,0]
    # fiber = fibermap[:,1]
    # xpos = fibermap[:,2].astype('float')
    # ypos = fibermap[:,3].astype('float') 
    # for i in range(3):
    #     for j in range(len(fibermap)):
    #         ax[i].text(xpos[j], ypos[j], fiber[j]+' '+bench[j], color='k', fontsize=4)


        
def run_extract(filepath):
    import os
    import ray
    import pkg_resources
    from   pathlib import Path        
    from   llamas_pyjamas.config import DATA_DIR    
    from   llamas_pyjamas.GUI.guiExtract import GUI_extract    
    ray.init(ignore_reinit_error=True)   
    
    # Get absolute path to llamas_pyjamas package to check the installation
    package_path = pkg_resources.resource_filename('llamas_pyjamas', '')
    package_root = os.path.dirname(package_path)
    
    print(f"Package path: {package_path}")
    print(f"Package root: {package_root}")    
    
    # Configure Ray runtime environment
    runtime_env = {
        "py_modules": [package_root],
        "env_vars": {"PYTHONPATH": f"{package_root}:{os.environ.get('PYTHONPATH', '')}"},
        "excludes": [
            str(Path(DATA_DIR) / "**"),  # Exclude DATA_DIR and all subdirectories
            "**/*.fits",                 # Exclude all FITS files anywhere
            "**/*.gz",                 # Exclude all tarballs files anywhere
            "**/*.zip",                 # Exclude all zip files anywhere
            "**/*.pkl",                  # Exclude all pickle files anywhere
            "**/.git/**",               # Exclude git directory
        ]
    }
    
    # Initialize Ray
    ray.shutdown()
    ray.init(runtime_env=runtime_env)    
    
    # Run extraction process
    GUI_extract(filepath)
    
def extract_fiber(exobj, LUT, aper):
    
    # Columns: bench, fiber, xpos, ypos
    fibermap = np.genfromtxt(LUT, delimiter='|', dtype=None, autostrip=True,
                             usecols=(1,2,5,6)).astype('str')[1:]
    xpos = fibermap[:,2].astype('float')
    ypos = fibermap[:,3].astype('float') 
    
    waves = []
    spectra = []
    for i, c in enumerate(['r', 'g', 'b']):
        dist = (xpos-aper[i][0])**2 + (ypos-aper[i][1])**2
        inds = np.argsort(dist)
        bench, fiber = fibermap[inds[0]][:2]    
        hduidx = np.nonzero(detector == bench)[0][0]*3 + i
        counts = exobj['extractions'][hduidx].counts
        color = exobj['extractions'][hduidx].channel
        if i == 2:
            counts = np.flipud(counts)    
        spectra.append(counts[int(fiber)])
        
        if c == 'g' or c == 'b':
            waves.append(np.loadtxt('wavecal_solutions/'+c+'_'+bench+'_'+fiber+'.txt'))
        else:
            waves.append(np.linspace(9800, 6900, 2048))

    return np.array(waves), np.array(spectra)
        
    
def extract_aper(exobj, LUT, aper, waves, sky, out_dir):
    import os
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from scipy.interpolate import interp1d
    
    # Columns: bench, fiber, xpos, ypos
    fibermap = np.genfromtxt(LUT, delimiter='|', dtype=None, autostrip=True,
                             usecols=(1,2,5,6)).astype('str')[1:]
    bench = fibermap[:,0]
    fiber = fibermap[:,1]
    xpos = fibermap[:,2].astype('float')
    ypos = fibermap[:,3].astype('float') 

    spectra = []
    
    for i, color in enumerate(['r', 'g', 'b']):
        dist = np.sqrt((xpos-aper[i][0])**2 + (ypos-aper[i][1])**2)
        inds = np.nonzero(dist < aperture_radius)[0]
        
        # Check wavelength solution exists (sometimes missing fibers e.g. 4A Fiber 298)
        if color == 'g' or color == 'b':
            inds = [ind for ind in inds if os.path.exists('wavecal_solutions/'+color+'_'+bench[ind]+'_'+fiber[ind]+'.txt')]
        
        aper_arr = []
        w_arr = []        
        for j in range(len(inds)):

            hduidx = np.nonzero(detector == bench[inds[j]])[0][0]*3 + i
            counts = exobj['extractions'][hduidx].counts
            if i == 2:
                counts = np.flipud(counts)
            counts = counts[int(fiber[inds[j]])]

            if color == 'g' or color == 'b':
                wave = np.loadtxt('wavecal_solutions/'+color+'_'+bench[inds[j]]+'_'+fiber[inds[j]]+'.txt')
            else:
                wave =  np.linspace(9800, 6900, 2048)
            interp_counts = interp1d(wave, counts, kind='linear',
                                            fill_value='extrapolate')
            
            counts = interp_counts(waves[i])
            skysub_counts = counts - sky[i]
            
            aper_arr.append(skysub_counts)
            w_arr.append(np.nansum(skysub_counts))

        aper_arr = np.array(aper_arr)
        w_arr = np.array(w_arr)
        w_arr = w_arr / np.sum(w_arr)        
        w_arr = np.repeat(np.expand_dims(w_arr, 1), 2048, axis=1)
        
        weighted_sum = np.nansum(aper_arr * w_arr, axis=0)  # Weighted sum
        normalization = np.nansum(w_arr * ~np.isnan(aper_arr), axis=0)  # Sum of valid weights
        
        avg_counts = weighted_sum / normalization  # Final weighted mean        

        spectra.append(avg_counts)

        fig, ax = plt.subplots(figsize=(12,5))
        ax.plot(waves[i], avg_counts, c=color)
        ax.set_xlabel('Wavelength (Å)')
        ax.set_ylabel('Counts')
        fig.savefig(out_dir+color+'_aper_add.png', dpi=300)

        nrows = 5
        sorted_inds = np.argsort(w_arr[:,0])[::-1]
        fig, ax = plt.subplots(figsize=(12,10), nrows=nrows, ncols=2, width_ratios=[1,0.2])
        plt.suptitle(color+' target aperture')
        for row, spec in enumerate(sorted_inds[:nrows]):
            # [j].plot(aper_arr[j], c=color)
            ax[row][0].plot(waves[i], aper_arr[spec], c=color)            
            ax[row][0].text(0.05, 0.95, 'Weight: '+str(np.round(w_arr[spec][0], 4)),
                        transform=ax[row][0].transAxes, ha='left', va='top')
        ax[2][0].set_ylabel('Counts')
        # ax[-1].set_xlabel('Pixel')
        ax[-1][0].set_xlabel('Wavelength (Å)')        
        inds_img = np.nonzero( (xpos>aper[i][0]-aperture_outersky) *\
                           (xpos<aper[i][0]+aperture_outersky) *\
                           (ypos>aper[i][1]-aperture_outersky) *\
                            (ypos<aper[i][1]+aperture_outersky))[0]
        xpos_img = xpos[inds_img]  
        ypos_img = ypos[inds_img]
        cnts_img = []
        for j in range(len(inds_img)):
            hduidx = np.nonzero(detector == bench[inds_img[j]])[0][0]*3 + i
            counts = exobj['extractions'][hduidx].counts
            if color == 'b':
                counts = np.flipud(counts)
            counts = counts[int(fiber[inds_img[j]])]
            cnts_img.append(np.nansum(counts))
        for row, spec in enumerate(sorted_inds[:nrows]):
            ax[row][1].scatter(xpos_img, ypos_img, c=cnts_img, cmap='viridis', marker='s', s=50)
            ax[row][1].plot([aper[i][0]], [aper[i][1]], '.m')
            circle = patches.Circle((aper[i][0],aper[i][1]), aperture_radius,
                                    ec='m', fc='none')
            ax[row][1].add_patch(circle)
            circle = patches.Circle((aper[i][0],aper[i][1]), aperture_innersky,
                                    ec='k', fc='none')
            ax[row][1].add_patch(circle)
            circle = patches.Circle((aper[i][0],aper[i][1]), aperture_outersky,
                                    ec='k', fc='none')
            ax[row][1].add_patch(circle)                
            rect = patches.Rectangle((xpos[inds[spec]] - 0.5, ypos[inds[spec]] - 0.5), 1, 1,
                                    linewidth=2, edgecolor='red', facecolor='none')
            ax[row][1].add_patch(rect)   
        # plt.subplots_adjust(hspace=0)
        fig.savefig(out_dir+color+'_aper.png', dpi=300)
        
    return spectra
        
    # inds = np.argsort(dist)
    # rbench, rfiber = fibermap[inds[0]][:2]      

def extract_sky(exobj, LUT, aper, out_dir):

    import os
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from scipy.interpolate import interp1d

    # Columns: bench, fiber, xpos, ypos
    fibermap = np.genfromtxt(LUT, delimiter='|', dtype=None, autostrip=True,
                             usecols=(1,2,5,6)).astype('str')[1:]
    bench = fibermap[:,0]
    fiber = fibermap[:,1]
    xpos = fibermap[:,2].astype('float')
    ypos = fibermap[:,3].astype('float')     

    spectra = []
    waves = []

    for i, color in enumerate(['r', 'g', 'b']):
        
        dist = np.sqrt((xpos-aper[i][0])**2 + (ypos-aper[i][1])**2)
        inds = np.nonzero( (dist<aperture_outersky) *\
                           (dist>aperture_innersky))[0]
        
        # Check wavelength solution exists
        if color == 'g' or color == 'b':
            inds = [ind for ind in inds if os.path.exists('wavecal_solutions/'+color+'_'+bench[ind]+'_'+fiber[ind]+'.txt')]

        # Get counts
        counts_arr = []
        for j in range(len(inds)):
            hduidx = np.nonzero(detector == bench[inds[j]])[0][0]*3 + i
            counts = exobj['extractions'][hduidx].counts
            if color == 'b':
                counts = np.flipud(counts)
            counts = counts[int(fiber[inds[j]])]

            if color == 'g' or color == 'b':
                wave = np.loadtxt('wavecal_solutions/'+color+'_'+bench[inds[j]]+'_'+fiber[inds[j]]+'.txt')
            else:
                wave = np.linspace(9800, 6900, 2048)
            
            if j == 0:
                wave_template = wave
            else:
                counts = interp1d(wave, counts, kind='nearest', fill_value='extrapolate')(wave_template)

            counts_arr.append(counts)

        # Interpolate sky spectrum to match science wavelength grid

        waves.append(wave_template)
        counts_arr = np.array(counts_arr)
        sky_counts = np.nanmedian(counts_arr, axis=0)
        spectra.append(sky_counts)

        fig, ax = plt.subplots(figsize=(12,5))
        ax.plot(wave_template, sky_counts, c=color)
        ax.set_xlabel('Wavelength (Å)')
        ax.set_ylabel('Counts')
        fig.savefig(out_dir+color+'_sky_median.png', dpi=300)

        nrows = 5
        fig, ax = plt.subplots(nrows=nrows, ncols=2, figsize=(12,10), width_ratios=[1, 0.2])
        for j in range(nrows):
            ax[j][0].plot(wave_template, counts_arr[j], c=color)
            ax[j][0].plot(wave_template, sky_counts, '--k', lw=1)
        ax[-1][0].set_xlabel('Wavelength (Å)')
        inds_img = np.nonzero( (xpos>aper[i][0]-aperture_outersky) *\
                           (xpos<aper[i][0]+aperture_outersky) *\
                           (ypos>aper[i][1]-aperture_outersky) *\
                            (ypos<aper[i][1]+aperture_outersky))[0]
        xpos_img = xpos[inds_img]  
        ypos_img = ypos[inds_img]
        cnts_img = []
        for j in range(len(inds_img)):
            hduidx = np.nonzero(detector == bench[inds_img[j]])[0][0]*3 + i
            counts = exobj['extractions'][hduidx].counts
            if color == 'b':
                counts = np.flipud(counts)
            counts = counts[int(fiber[inds_img[j]])]
            cnts_img.append(np.nansum(counts))
        for j in range(nrows):
            ax[j][1].scatter(xpos_img, ypos_img, c=cnts_img, cmap='viridis', marker='s', s=50)
            ax[j][1].plot([aper[i][0]], [aper[i][1]], '.m')
            circle = patches.Circle((aper[i][0],aper[i][1]), aperture_radius,
                                    ec='m', fc='none')
            ax[j][1].add_patch(circle)
            circle = patches.Circle((aper[i][0],aper[i][1]), aperture_innersky,
                                    ec='k', fc='none')
            ax[j][1].add_patch(circle)
            circle = patches.Circle((aper[i][0],aper[i][1]), aperture_outersky,
                                    ec='k', fc='none')
            ax[j][1].add_patch(circle)                
            rect = patches.Rectangle((xpos[inds[j]] - 0.5, ypos[inds[j]] - 0.5), 1, 1,
                                    linewidth=2, edgecolor='red', facecolor='none')
            ax[j][1].add_patch(rect)            
        # plt.subplots_adjust(hspace=0)
        fig.savefig(out_dir+color+'_sky.png', dpi=300)

    return waves, spectra
        

def est_continuum(wave, flux, bins=100):
    from scipy.interpolate import interp1d
    from scipy.stats import binned_statistic
    import matplotlib.pyplot as plt

    ulines = [4340.472, 4101.734, 3970.075, 3889.064, 3835.397]
    uwid   = [45,       45,       30,       25,       15]
    glines = [6562.79,  4861.35]
    gwid   = [60,       45]
    rlines = []
    rwid   = []

    # Remove Balmer series 
    continuum_inds = np.ones(len(wave), dtype='bool')
    for line, width in zip(rlines+glines+ulines,rwid+gwid+uwid):
        continuum_inds *= ~((wave > line-width) * (wave < line+width))
    continuum_wave = wave[continuum_inds]
    continuum_flux = flux[continuum_inds]

    # Smooth spectrum to approximate the continuum
    xmap = binned_statistic(continuum_wave, continuum_wave, bins=bins).statistic
    ymap = binned_statistic(continuum_wave, continuum_flux, bins=bins,
                            statistic='median').statistic 
    inds = np.nonzero(~np.isnan(ymap))
    xmap, ymap = xmap[inds], ymap[inds]
    contmap = interp1d(xmap, ymap, fill_value="extrapolate")
    
    continuum = contmap(wave)
    
    # # Deal with negative values
    # inds = np.nonzero(continuum < 1)
    # continuum[inds] = 1
    
    # plt.figure(figsize=(12,4))
    # plt.title('Continuum estimation')
    # plt.plot(wave, flux, '-k', label='Raw data')
    # plt.plot(xmap, ymap, '.m', label='Binned data')
    # plt.plot(wave, continuum(wave), '--b', label='Continuum fit')
    # plt.legend()
    # plt.xlabel('Wavelength (Å)')
    # plt.ylabel('Counts')
    
    return continuum
        
    
def est_sensfunc():
    from scipy.stats import binned_statistic
    import matplotlib.pyplot as plt
    from scipy.interpolate import interp1d
    from numpy.polynomial.polynomial import Polynomial
    from scipy.signal import savgol_filter
    
    hstspec = np.loadtxt('/Users/emma/projects/LLAMAS_UTILS/LLAMAS_UTILS/ffeige110.dat')
    
    rspec = np.loadtxt('/Users/emma/Desktop/work/250214/F110_LLAMAS_2024-11-28T01_22_09.108_red_counts.txt')
    gspec = np.loadtxt('/Users/emma/Desktop/work/250214/F110_LLAMAS_2024-11-28T01_22_09.108_green_counts.txt')
    uspec = np.loadtxt('/Users/emma/Desktop/work/250214/F110_LLAMAS_2024-11-28T01_22_09.108_blue_counts.txt')
    
    # # Deal with negative values
    # inds = np.nonzero(rspec[:,1] < 1)
    # rspec[:,1][inds] = 1    
    # inds = np.nonzero(gspec[:,1] < 1)
    # gspec[:,1][inds] = 1   
    # inds = np.nonzero(uspec[:,1] < 1)
    # uspec[:,1][inds] = 1       
    
    hstmap = interp1d(hstspec[:,0], hstspec[:,1], fill_value="extrapolate")

    # Smooth HST spectra to approximate the continuum
    r_true = est_continuum(rspec[:,0], hstmap(rspec[:,0]))
    g_true = est_continuum(gspec[:,0], hstmap(gspec[:,0]))
    u_true = est_continuum(uspec[:,0], hstmap(uspec[:,0]))
    
    # Smooth LLAMAS spectra to approximate the continuum
    r_continuum = est_continuum(rspec[:,0], rspec[:,1])
    g_continuum = est_continuum(gspec[:,0], gspec[:,1])
    u_continuum = est_continuum(uspec[:,0], uspec[:,1])
    
    # Calculate instrument response
    r_response = r_true / r_continuum
    g_response = g_true / g_continuum
    u_response = u_true / u_continuum
    
    # Smooth the response function
    r_response = savgol_filter(r_response, 151, 3)
    g_response = savgol_filter(g_response, 151, 3)
    u_response = savgol_filter(u_response, 151, 3)
    
    
    # r_response = hstmap(rspec[:,0]) / r_continuum
    # g_response = hstmap(gspec[:,0]) / g_continuum
    # u_response = hstmap(uspec[:,0]) / u_continuum    
    
    # # Normalize the LLAMAS spectra
    # r_norm = rspec[:,1] / r_continuum
    # g_norm = gspec[:,1] / g_continuum
    # u_norm = uspec[:,1] / u_continuum    

    # Calculate instrument response
    # r_response = hstmap(rspec[:,0]) / r_norm 
    # g_response = hstmap(gspec[:,0]) / g_norm
    # u_response = hstmap(uspec[:,0]) / u_norm
    
    # rcnt = np.nansum(rspec[:,1])
    # gcnt = np.nansum(gspec[:,1])
    # ucnt = np.nansum(uspec[:,1])    
    
    # -------------------------------------------------------------------------
    
    fig, ax = plt.subplots(figsize=(20, 7), nrows=2, ncols=3)
    plt.suptitle('Sensitivity in red channel')
    
    ax[0][2].plot(rspec[:,0], rspec[:,1], '-r', label='Raw LLAMAS spectrum')
    ylim = ax[0][0].get_ylim()
    ax[0][2].plot(rspec[:,0], r_response*1000, '-k', label='scaled Sensitivity')
    ax[0][2].set_ylim(ylim)
    ax[1][2].plot(rspec[:,0], hstmap(rspec[:,0])+500, '-k', label='HST Spectrum')
    ax[1][2].plot(rspec[:,0], r_response*rspec[:,1], '--r',
                  label='Flux-calibrated LLAMAS spectrum')
      
    ax[0][1].plot(gspec[:,0], gspec[:,1], '-g', label='Raw LLAMAS spectrum')
    ylim = ax[0][1].get_ylim()
    ax[0][1].plot(gspec[:,0], g_response*1000, '-k', label='scaled Sensitivity')
    ax[0][1].set_ylim(ylim)
    ax[1][1].plot(gspec[:,0], hstmap(gspec[:,0])+500, '-k', label='HST Spectrum')
    ax[1][1].plot(gspec[:,0], g_response*gspec[:,1], '--g',
                  label='Flux-calibrated LLAMAS spectrum')

    ax[0][0].plot(uspec[:,0], uspec[:,1], '-b', label='Raw LLAMAS spectrum')
    ylim = ax[0][2].get_ylim()
    ax[0][0].plot(uspec[:,0], u_response*1000, '-k', label='scaled Sensitivity')
    ax[0][0].set_ylim(ylim)
    ax[1][0].plot(uspec[:,0], hstmap(uspec[:,0])+500, '-k', label='HST Spectrum')
    ax[1][0].plot(uspec[:,0], u_response*uspec[:,1], '--b', 
                  label='Flux-calibrated LLAMAS spectrum')    


    ax[0][0].set_ylabel('Counts')
    ax[0][0].legend()
    ax[1][0].set_ylabel(r'Flux (erg cm$^{-2}$ s$^{-1}$ Å$^{-1}$ $\times$ 10$^{16}$)')
    ax[1][0].set_xlabel('Wavelength (Å)')
    ax[1][0].legend()
    ax[0][1].legend()
    ax[1][1].set_xlabel('Wavelength (Å)')
    ax[1][1].legend()
    ax[0][2].legend()
    ax[1][2].set_xlabel('Wavelength (Å)')    
    ax[1][2].legend()
    
    plt.subplots_adjust(hspace=0, left=0.05, right=0.99)
    
    # plt.figure(figsize=(12,4))
    # plt.title('HST Spectrum of Feige 110')
    # # plt.plot(hstspec[:,0], hstspec[:,1], '-k', label='HST Spectrum')
    # plt.plot(rspec[:,0], hstmap(rspec[:,0]), '-.r', label='HST Spectrum')
    # plt.plot(rspec[:,0], r_true, '--k', label='Continuum fit')
    # plt.plot(gspec[:,0], hstmap(gspec[:,0]), '-.g')
    # plt.plot(gspec[:,0], g_true, '--k')
    # plt.plot(uspec[:,0], hstmap(uspec[:,0]), '-.b')
    # plt.plot(uspec[:,0], u_true, '--k')
    # plt.xlabel('Wavelength (Å)')
    # plt.ylabel(r'Flux (erg cm$^{-2}$ s$^{-1}$ Å$^{-1}$ $\times$ 10$^{16}$)')
    # plt.legend()
    
    # fig, ax = plt.subplots(nrows=3, figsize=(12,7))
    # plt.suptitle('Flux calibration of Feige 110')
    
    # inds = np.nonzero((hstspec[:,0]>np.min(rspec[:,0])) * \
    #                   (hstspec[:,0]<np.max(rspec[:,0])))
    # ax[2].plot(hstspec[:,0][inds], hstspec[:,1][inds], '-k', label='HST Spectrum')
    # ax[2].plot(rspec[:,0], r_response*rspec[:,1], '-.r',
    #            label='Flux-calibrated LLAMAS spectrum')
    # ax[2].set_xlabel('Wavelength (Å)')
    # ax[2].legend()

    # inds = np.nonzero((hstspec[:,0]>np.min(gspec[:,0])) * \
    #                   (hstspec[:,0]<np.max(gspec[:,0])))
    # ax[1].plot(hstspec[:,0][inds], hstspec[:,1][inds], '-k', label='HST Spectrum')
    # ax[1].plot(gspec[:,0], g_response*gspec[:,1], '-.g',
    #            label='Flux-calibrated LLAMAS spectrum')
    # ax[1].set_ylabel(r'Flux (erg cm$^{-2}$ s$^{-1}$ Å$^{-1}$ $\times$ 10$^{16}$)')    
    # ax[1].legend()

    # inds = np.nonzero((hstspec[:,0]>np.min(uspec[:,0])) * \
    #                   (hstspec[:,0]<np.max(uspec[:,0])))
    # ax[0].plot(hstspec[:,0][inds], hstspec[:,1][inds], '-k', label='HST Spectrum')
    # ax[0].plot(uspec[:,0], u_response*uspec[:,1], '-.b',
    #            label='Flux-calibrated LLAMAS spectrum') 
    # ax[0].legend()
    # plt.subplots_adjust(hspace=0)
    
    # fig, ax = plt.subplots(nrows=3, figsize=(12,7))
    # plt.suptitle('Sensitivity function')
    # ax[2].plot(rspec[:,0], r_response, '-k')
    # ax[2].set_xlabel('Wavelength (Å)')
    # ax[1].plot(gspec[:,0], g_response, '-k')
    # ax[1].set_ylabel('Instrument response')
    # ax[0].plot(uspec[:,0], u_response, '-k')
    
    # fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(12,7))
    # plt.suptitle('Normalization of LLAMAS Spectrum of Feige 110')
    
    # ax[2][0].plot(rspec[:,0], rspec[:,1], '-r')
    # ax[2][0].plot(rspec[:,0], r_continuum, '-.k')
    # ax[2][1].plot(rspec[:,0], r_norm, '-r')
    # ax[2][0].set_xlabel('Wavelength (Å)')
    # ax[2][1].set_xlabel('Wavelength (Å)')
    
    # ax[1][0].plot(gspec[:,0], gspec[:,1], '-g')
    # ax[1][0].plot(gspec[:,0], g_continuum, '-.k')
    # ax[1][1].plot(gspec[:,0], g_norm, '-g')
    # ax[1][0].set_ylabel('Counts')
    # ax[1][1].set_ylabel('Relative Flux')

    # ax[0][0].plot(uspec[:,0], uspec[:,1], '-b')
    # ax[0][0].plot(uspec[:,0], u_continuum, '-.k')
    # ax[0][1].plot(uspec[:,0], u_norm, '-b')
    # ax[0][1].set_title('Normalized Spectrum')
    # ax[0][0].set_title('Raw Spectrum + Continuum fit')

    # plt.subplots_adjust(hspace=0)      
    
    
    
#     xmap = llamascont(rspec[:,0])
#     ymap = hstcont(rspec[:,0])
#     sensfunc = interp1d(xmap, ymap, fill_value="extrapolate")
    
#     plt.figure()
#     plt.plot(rspec[:,0], sensfunc(rspec[:,1]))
#     plt.plot(rspec[:,0], hstcont(rspec[:,0]), '-', label='continuum')    

    
    return r_response, g_response, u_response # , rcnt, gcnt, ucnt

def wavecal():
    from scipy.interpolate import interp1d
    
    waves = []
    bands = [[6900, 9800], [4750, 6900], [3500, 4750]]
    
    red_wave = np.linspace(bands[0][1], bands[0][0], 2048)       
    waves.append(red_wave)

    
    green_wave = np.linspace(bands[1][1], bands[1][0], 2048)
    xmap = np.array([4948, 6426, 4802.3])
    ymap = np.array([4861.35, 6562.79, 4713])
    interp_func = interp1d(xmap, ymap, kind='linear', fill_value="extrapolate")
    green_wave = interp_func(green_wave)
    waves.append(green_wave)
    
    blue_wave = np.linspace(bands[2][1], bands[2][0], 2048)
    xmap = np.array([4253.4, 4056.8, 3953.2, 3886.7, 4364])
    ymap = np.array([4340.472, 4101.734, 3970.075, 3889.064, 4471])
    interp_func = interp1d(xmap, ymap, kind='linear', fill_value="extrapolate")
    blue_wave = interp_func(blue_wave)    
    waves.append(blue_wave)
     
    return np.array(waves)

def fluxcal(waves, spectra):
    from scipy.stats import binned_statistic
    from scipy.interpolate import interp1d
    import matplotlib.pyplot as plt
    # from scipy.interpolate import make_smoothing_spline
    
    rfunc, gfunc, ufunc = est_sensfunc()
    
    calib_spectra = []
    
    for i, color in enumerate(['red','green','blue']):
        wave, flux = waves[i], spectra[i]
        
        
        # continuum = est_continuum(wave, flux)
        # norm_flux = flux / continuum     
        
        if color == 'red':
        #     scaling_factor = np.nansum(flux) / rcnt
            response = rfunc
            
        elif color == 'green':
        #     scaling_factor = np.nansum(flux) / gcnt
            response = gfunc
        elif color == 'blue':
        #     scaling_factor = np.nansum(flux) / ucnt
            response = ufunc
        
        # cal_flux = norm_flux * response * scaling_factor
        cal_flux = flux * response
        calib_spectra.append(cal_flux)
        
        # # Remove nans
        # num_inds = np.nonzero(~np.isnan(flux))
        # wave, flux = wave[num_inds], flux[num_inds]    
        
        # # Estimate continuum
        # if color == 'blue':
        #     inds = np.nonzero(~((wave>4245)*(wave<4431)) * ~((wave>4032)*(wave<4152))*
        #                       ~((wave>3933)*(wave<3966)) * ~((wave<3898)*(wave>3850.6))*
        #                       ~((wave>4200)*(wave<4241)))
        #     wave_cont, flux_cont = wave[inds], flux[inds]
            
        #     sensfunc = ufunc
            
        # if color == 'green':
        #     inds = np.nonzero(~((wave>4811)*(wave<4918)))
        #     wave_cont, flux_cont = wave[inds], flux[inds]
            
        #     sensfunc = gfunc
            
        # if color == 'red':
        #     inds = np.nonzero(~((wave>9020)*(wave<9145)))
        #     wave_cont, flux_cont = wave[inds], flux[inds]     
            
        #     sensfunc = rfunc
        
        # xmap = binned_statistic(wave_cont, wave_cont, bins=60).statistic
        # ymap = binned_statistic(wave_cont, flux_cont, bins=60, statistic='median').statistic 
        # inds = np.nonzero(~np.isnan(ymap))
        # xmap, ymap = xmap[inds], ymap[inds]
        # # continuum = make_smoothing_spline(xmap, ymap)
        # continuum = interp1d(xmap, ymap, fill_value="extrapolate")   
        
        # norm_flux = flux / continuum(wave) * sensfunc(wave)    
        
        # reshape_flux = np.ones(spectra[i].shape) * np.nan
        # reshape_flux[num_inds]= norm_flux
        
        # calib_flux = sensfunc(flux)
        
        # reshape_flux = np.ones(spectra[i].shape) * np.nan
        # reshape_flux[num_inds]= calib_flux
        
        # calib_spectra.append(reshape_flux)
        
        # Plot channel
        # plt.figure()
        # plt.plot(wave, flux, '-', color=color)
        # plt.plot(xmap, ymap, '.k')
        # plt.plot(wave, continuum(wave), '-k')
        # plt.xlabel('Wavelength [Angstrom]')
        # plt.ylabel('Counts')
    
        
    return np.array(calib_spectra)

    