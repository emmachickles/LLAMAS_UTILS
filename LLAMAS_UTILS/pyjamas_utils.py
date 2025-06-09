import numpy as np

aperture_radius = 1.5
aperture_innersky = 2.5
aperture_outersky = 3.5
detector = np.array(['1A', '1B', '2A', '2B', '3A', '3B', '4A', '4B'])

def load_arc_pkl():
    from llamas_pyjamas.config import BASE_DIR
    import pickle, os

    arc_filepath = os.path.join(BASE_DIR, 'Arc', 'LLAMAS_reference_arc.pkl')
    # arc_filepath = os.path.join(BASE_DIR, 'Arc', 'LLAMAS Reference Arc Pre Blue Realign.pkl')
    with open(arc_filepath, 'rb') as f:
        arc = pickle.load(f)
    return arc

def load_LUT():
    from llamas_pyjamas.config import LUT_DIR
    import os
    from astropy.table import Table 
    fibre_map_path = os.path.join(LUT_DIR, 'LLAMAS_FiberMap_rev02.dat')
    fibermap_lut = Table.read(fibre_map_path, format='ascii.fixed_width')
    return fibermap_lut

def extract_fiber_by_fibnum(exobj, arc, color, bench, fiber, telluric_correction=False):

    if color == 'r':
        color = 'red'
    if color == 'g':
        color = 'green'
    if color == 'b':
        color = 'blue'

    metadata_channel = np.array([chan['channel'] for chan in exobj['metadata']])
    metadata_bench = np.array([chan['bench'] for chan in exobj['metadata']]).astype('int')
    metadata_side = np.array([chan['side'] for chan in exobj['metadata']])

    hduidx = np.nonzero( (metadata_channel==color) * \
                        (metadata_bench==int(bench[0])) * \
                        (metadata_side==bench[1]))[0][0]
    # hduidx = np.nonzero(detector == bench)[0][0]*3 + color

    counts = exobj['extractions'][hduidx].counts
    counts = counts[int(fiber)]

    waves = arc['extractions'][hduidx].wave[int(fiber)]

    if telluric_correction:
        waves, _ = refine_wavelength_with_telluric(waves, counts)

    return waves, counts


def get_brightest_pix(whitelight):
    aper = []
    
    for i in [5,3,1]:
        
        img = np.fliplr( whitelight[5].data )
        maxcnt = np.nanmax(img)
        xy = np.where(img == maxcnt)
        xy = [xy[1][0]/1.5, xy[0][0]/1.5]
        aper.append(xy)
    
    return aper

def plot_whitelight(whitelight, x, y):

    from astropy.io import fits
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig, ax = plt.subplots(figsize=(20,5), ncols=3)
    ax[0].set_title(whitelight[5].header['EXTNAME'])
    ax[0].imshow(whitelight[5].data, origin='lower', aspect='equal',
                 extent=[0, 80/1.5, 0, 80/1.5])
    ax[1].set_title(whitelight[3].header['EXTNAME'])
    ax[1].imshow(whitelight[3].data, origin='lower', aspect='equal',
                 extent=[0, 80/1.5, 0, 80/1.5])
    ax[2].set_title(whitelight[1].header['EXTNAME'])
    ax[2].imshow(whitelight[1].data, origin='lower', aspect='equal',
                 extent=[0, 80/1.5, 0, 80/1.5])
    for i in range(3):        
        ax[i].plot([x], [y], '.m')
        circle = patches.Circle((x, y), aperture_radius,
                                ec='m', fc='none')
        ax[i].add_patch(circle)
        circle = patches.Circle((x, y), aperture_innersky,
                                ec='k', fc='none')
        ax[i].add_patch(circle)
        circle = patches.Circle((x, y), aperture_outersky,
                                ec='k', fc='none')
        ax[i].add_patch(circle)    

        ax[i].set_xlabel('xpos')
        ax[i].set_ylabel('ypos')     

def plot_fiber_intensity_map(exobj, fibermap, arc, x, y, out_dir):

    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig, ax = plt.subplots(figsize=(20,5), ncols=3)

    for i, color in enumerate(['r', 'g', 'b']):
        
        sumcnts = [] 
        xpos_cnts = []
        ypos_cnts = []
        for fib in range(len(fibermap)):
            try:
                _, counts = extract_fiber_by_fibnum(exobj, arc, color, fibermap['bench'][fib], fibermap['fiber'][fib])
                sumcnts.append( np.nansum(counts) )
                xpos_cnts.append(fibermap['xpos'][fib])
                ypos_cnts.append(fibermap['ypos'][fib])
            except:
                pass

        ax[i].set_title(color)
        ax[i].scatter(xpos_cnts, ypos_cnts, c=sumcnts, cmap='viridis', marker='s', s=10)
        ax[i].plot([x], [y], '.m')
        circle = patches.Circle((x, y), aperture_radius,
                                ec='m', fc='none')
        ax[i].add_patch(circle)
        circle = patches.Circle((x, y), aperture_innersky,
                                ec='k', fc='none')
        ax[i].add_patch(circle)
        circle = patches.Circle((x, y), aperture_outersky,
                                ec='k', fc='none')
        ax[i].add_patch(circle)
        ax[i].set_xlabel('xpos')
        ax[i].set_ylabel('ypos')  
    fig.savefig(out_dir+'fiber_intensity.png', dpi=300)        
        
def get_local_fiber_intensity_data(exobj, fibermap, arc, x, y, color, half_width=6.):
    """
    Returns a structured numpy array with fields:
    xpos, ypos, sumcnt, bench, fiber for fibers within a square region.
    """
    dtype = [
        ('xpos', float),
        ('ypos', float),
        ('sumcnt', float),
        ('bench', fibermap['bench'].dtype),
        ('fiber', fibermap['fiber'].dtype)
    ]
    map_data = []

    # Select fibers within the square region
    in_square = np.where(
        (fibermap['xpos'] >= x - half_width) &
        (fibermap['xpos'] <= x + half_width) &
        (fibermap['ypos'] >= y - half_width) &
        (fibermap['ypos'] <= y + half_width)
    )[0]

    for fib in in_square:
        _, counts = extract_fiber_by_fibnum(exobj, arc, color, fibermap['bench'][fib], fibermap['fiber'][fib])
        map_data.append((
            fibermap['xpos'][fib],
            fibermap['ypos'][fib],
            np.nansum(counts),
            fibermap['bench'][fib],
            fibermap['fiber'][fib]
        ))

    return np.array(map_data, dtype=dtype)

def plot_local_fiber_intensity_on_axis(ax, map_data, x, y, color, half_width=6.):
    import matplotlib.patches as patches

    ax.set_title(color)
    xpos_cnts = map_data['xpos']
    ypos_cnts = map_data['ypos']
    sumcnts = map_data['sumcnt']
    benches = map_data['bench']
    fibers = map_data['fiber']

    sc = ax.scatter(xpos_cnts, ypos_cnts, c=sumcnts, cmap='viridis', marker='s', s=30)
    for xpos, ypos, bench, fiber in zip(xpos_cnts, ypos_cnts, benches, fibers):
        ax.text(xpos, ypos, f'{bench}\n{fiber}', color='black', fontsize=3, ha='center', va='center')
    ax.plot([x], [y], '.m')
    circle = patches.Circle((x, y), aperture_radius, ec='m', fc='none')
    ax.add_patch(circle)
    circle = patches.Circle((x, y), aperture_innersky, ec='k', fc='none')
    ax.add_patch(circle)
    circle = patches.Circle((x, y), aperture_outersky, ec='k', fc='none')
    ax.add_patch(circle)
    ax.set_xlim(x - half_width, x + half_width)
    ax.set_ylim(y - half_width, y + half_width)
    ax.set_xlabel('xpos')
    ax.set_ylabel('ypos')

def plot_local_fiber_intensity_single_color(exobj, fibermap, arc, x, y, color, ax, half_width=6.):
    map_data = get_local_fiber_intensity_data(
        exobj, fibermap, arc, x, y, color, half_width=half_width
    )
    plot_local_fiber_intensity_on_axis(
        ax, map_data, x, y, color, half_width=half_width
    )


def plot_local_fiber_intensity(exobj, fibermap, arc, x, y, out_dir, half_width=6.):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(figsize=(8, 3), ncols=3)
    for i, color in enumerate(['r', 'g', 'b']):
        plot_local_fiber_intensity_single_color(
            exobj, fibermap, arc, x, y, color, axes[i], half_width=half_width
        )
    fig.tight_layout()
    fig.savefig(out_dir + 'fiber_intensity_loc.png', dpi=300)

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
    
def extract_fiber_by_pos(exobj, fibermap, arc, x, y):
    
    spec = {}
    for i, c in enumerate(['r', 'g', 'b']):
        dist = (fibermap['xpos']-x)**2 + (fibermap['ypos']-y)**2
        inds = np.argsort(dist)
        bench, fiber = fibermap[inds[0]]['bench'], fibermap[inds[0]]['fiber']
        print(bench, fiber)
        hduidx = np.nonzero(detector == bench)[0][0]*3 + i
        counts = exobj['extractions'][hduidx].counts
        waves = arc['extractions'][hduidx].wave

        spec[c] = {'wave': waves[int(fiber)], 'spectrum': counts[int(fiber)]}

    return spec
        
def extract_aper(exobj, fibermap, arc, x_sci, y_sci):

    from scipy.interpolate import interp1d

    # Identify fibers in the science aperture
    dist = np.sqrt((fibermap['xpos']-x_sci)**2 + (fibermap['ypos']-y_sci)**2)
    aper_inds = np.nonzero((dist < aperture_radius))[0]
    print(len(aper_inds), 'fibers in the science aperture')

    # Match wavelength grid 
    spec = {}
    for i, c in enumerate(['r', 'g', 'b']):
        aper_cnts = []
        aper_wt = []
        all_spectra = []
        for j, fib in enumerate(aper_inds):
            wave, cnts = extract_fiber_by_fibnum(exobj, arc, c, fibermap['bench'][fib], fibermap['fiber'][fib])
            if j == 0:
                wave_template = wave
            else:
                interp_func = interp1d(wave, cnts, kind='linear', fill_value=np.nan, bounds_error=False)
                cnts = interp_func(wave_template)
            aper_cnts.append(cnts)
            aper_wt.append(np.nansum(cnts)**2)
            all_spectra.append(cnts)
        aper_cnts = np.array(aper_cnts)
        aper_wt = np.array(aper_wt)
        aper_wt = aper_wt / np.nansum(aper_wt)  # Normalize weights
        weighted_mean = np.nansum(aper_cnts * aper_wt[:, None], axis=0)

        spec[c] = {
            'wave': wave_template,
            'spectrum': weighted_mean,
            'weight': aper_wt,
            'bench': fibermap['bench'][aper_inds],
            'fiber': fibermap['fiber'][aper_inds],
            'all_spectra': np.array(all_spectra)
        }

    return spec

          
def extract_sky(exobj, fibermap, arc, x_sci, y_sci, spec_sci, bench_sci=True):
    """Extract the sky spectrum from fibers in the sky annulus around the science aperture.
    Currently only uses sky fibers from the bench that the science target falls in."""

    from scipy.interpolate import interp1d

    # Identify fibers in the sky annulus
    dist = np.sqrt((fibermap['xpos']-x_sci)**2 + (fibermap['ypos']-y_sci)**2)
    annul_inds = np.nonzero((dist > aperture_innersky) * (dist < aperture_outersky))[0]            
    print(len(annul_inds), 'fibers in the sky annulus')
    if bench_sci:
        bench_sci = fibermap['bench'][np.argmin(dist)]
        # Filter annul_inds to only include fibers from the same bench as the science target
        annul_inds = annul_inds[fibermap['bench'][annul_inds] == bench_sci]

    # Interpolate sky spectra to match science wavelength grid
    spec_sky = {}
    for i, c in enumerate(['r', 'g', 'b']):
        annul_cnts = []
        for j, fib in enumerate(annul_inds):
            wave, cnts = extract_fiber_by_fibnum(exobj, arc, c, fibermap['bench'][fib], fibermap['fiber'][fib])
            interp_func = interp1d(wave, cnts, kind='linear', fill_value=np.nan, bounds_error=False)
            annul_cnts.append(interp_func(spec_sci[c]['wave']))
        annul_cnts = np.array(annul_cnts)
        spec_sky[c] = {
            'wave': spec_sci[c]['wave'],
            'spectrum': np.nanmedian(annul_cnts, axis=0),
            'bench': fibermap['bench'][annul_inds],
            'fiber': fibermap['fiber'][annul_inds],
            'all_spectra': annul_cnts
        }
    return spec_sky

def highlight_patch(ax, fibermap, b, f):
    import matplotlib.patches as patches
    idx2 = np.where((fibermap['bench'] == b) & (fibermap['fiber'] == f))[0]
    xh, yh = fibermap['xpos'][idx2], fibermap['ypos'][idx2]
    rect = patches.Rectangle((xh - 0.5, yh - 0.5), 1, 1, linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rect)

def plot_aperture_fibers(exobj, fibermap, arc, x, y, spec, out_dir):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import matplotlib.gridspec as gridspec

    colors = ['r', 'g', 'b']
    for cidx, color in enumerate(colors):
        # Always compute local_map_data for this color
        local_map_data = get_local_fiber_intensity_data(
            exobj, fibermap, arc, x, y, color, half_width=6.
        )

        # Set width ratios: left column 3x wider than right
        fig = plt.figure(figsize=(16, 15))  # wider figure for new ratio
        gs = gridspec.GridSpec(6, 2, width_ratios=[3, 1], hspace=0.15, wspace=0.08)

        axes = np.empty((6, 2), dtype=object)
        for i in range(6):
            for j in range(2):
                axes[i, j] = fig.add_subplot(gs[i, j])

        spec_c = spec[color]
        wave = spec_c['wave']
        mean_spectrum = spec_c['spectrum']
        benchs = spec_c['bench']
        fibers = spec_c['fiber']
        all_spectra = spec_c['all_spectra']

        # Handle missing 'weight' key
        if 'weight' in spec_c:
            weights = spec_c['weight']
        else:
            # Use uniform weights if missing
            weights = np.ones(len(benchs)) / len(benchs)

        top_inds = np.argsort(weights)[-5:][::-1]  # indices of 5 highest weighted fibers

        # Plot mean spectrum
        ax_spec = axes[0, 0]
        n_fibers = len(benchs)
        ax_spec.plot(wave, mean_spectrum, color=color, label=f'{color} mean ({n_fibers} fibers)')
        ax_spec.set_xlabel('Wavelength')
        ax_spec.set_ylabel('Counts')
        ax_spec.legend()
        ax_spec.set_title('')  # Remove title

        # Plot local fiber intensity map for mean spectrum (no highlight)
        ax_map = axes[0, 1]
        plot_local_fiber_intensity_on_axis(
            ax_map, local_map_data, x, y, color, half_width=6.
        )
        ax_map.set_title('')  # Remove title
        ax_map.set_aspect('equal', adjustable='box')

        # Plot top 5 fibers
        for j, idx in enumerate(top_inds):
            b = benchs[idx]
            f = fibers[idx]
            w = weights[idx]
            # Use all_spectra instead of extracting again
            cnts_f = all_spectra[idx]
            axes[j + 1, 0].plot(wave, cnts_f, color=color, label=f'Bench {b}, Fiber {f} (weight={w:.2f})')
            axes[j + 1, 0].set_xlabel('Wavelength')
            axes[j + 1, 0].set_ylabel('Counts')
            axes[j + 1, 0].legend()
            axes[j + 1, 0].set_title('')  # Remove title

            # Plot local fiber intensity map with highlight
            plot_local_fiber_intensity_on_axis(
                axes[j + 1, 1], local_map_data, x, y, color, half_width=6.
            )
            highlight_patch(axes[j + 1, 1], fibermap, b, f)
            axes[j + 1, 1].set_title('')  # Remove title
            axes[j + 1, 1].set_aspect('equal', adjustable='box')

        fig.tight_layout(pad=0.5)
        fig.savefig(out_dir + f'fiber_{color}.png', dpi=300)
        plt.show()

def sky_subtraction(spec, spec_sky):
    """
    Subtracts the mean sky spectrum from each individual science fiber spectrum,
    then recomputes the weighted mean science spectrum. Returns a new spec_skysub
    dictionary in the same format as spec.
    Assumes spec[color]['all_spectra'] is present.
    """
    spec_skysub = {}
    for color in ['r', 'g', 'b']:
        wave = spec[color]['wave']
        benchs = spec[color]['bench']
        fibers = spec[color]['fiber']
        weights = spec[color]['weight']
        all_spectra = spec[color]['all_spectra']
        n_fibers = len(benchs)

        # Subtract mean sky from each fiber's science spectrum
        sky_spec = spec_sky[color]['spectrum']
        aper_cnts = all_spectra - sky_spec  # shape: (n_fibers, n_wave)

        aper_wt = np.array(weights)
        aper_wt = aper_wt / np.nansum(aper_wt)  # Normalize weights

        # Weighted mean of sky-subtracted spectra
        weighted_mean = np.nansum(aper_cnts * aper_wt[:, None], axis=0)

        spec_skysub[color] = {
            'wave': wave,
            'spectrum': weighted_mean,
            'weight': aper_wt,
            'bench': benchs,
            'fiber': fibers,
            'all_spectra': aper_cnts
        }

    return spec_skysub

def identify_emission_lines_with_pypeit(wave, cnts, sigdetect=5.0):
    """
    Identify emission lines in a spectrum using Pypeit's detect_lines,
    and return the detection parameters directly.

    Parameters
    ----------
    wave : array-like
        Wavelength array.
    cnts : array-like
        Flux/counts array.
    sigdetect : float
        Detection threshold for Pypeit's detect_lines.

    Returns
    -------
    lines : list of dict
        Each dict contains 'center', 'amplitude', 'amplitude_contsub', 'sigma', 'offset',
        'center_var', 'reliable', 'nsig', and 'fit_success'.
    """
    import numpy as np
    from pypeit.core import arc
    import warnings
    from astropy.utils.exceptions import AstropyWarning

    warnings.filterwarnings("ignore", category=AstropyWarning)    

    # Use Pypeit's detect_lines to find emission line peaks and their properties
    result = arc.detect_lines(cnts, sigdetect=sigdetect)

    # Unpack results
    tcent = result[2]
    tampl = result[0]
    tampl_cont = result[1]
    twid = result[3]
    center_var = result[4]
    w = result[5]
    nsig = result[7]

    lines = []
    for i in range(len(tcent)):
        # Map centroid to wavelength
        center_wave = wave[int(round(tcent[i]))] if tcent[i] >= 0 and tcent[i] < len(wave) else np.nan
        lines.append({
            'center': center_wave,
            'amplitude': tampl[i],
            'amplitude_contsub': tampl_cont[i],
            'sigma': abs(twid[i]),
            'offset': 0.0,  # Not available from detect_lines
            'center_var': center_var[i],
            'reliable': (i in w),
            'nsig': nsig[i],
            'fit_success': nsig[i] > 0
        })

    return lines

def plot_emission_line_fits(wave, cnts, lines, out_dir):
    """
    Plot the spectrum and best-fit Gaussians for identified emission lines.

    Parameters
    ----------
    wave : array-like
        Wavelength array.
    cnts : array-like
        Flux/counts array.
    lines : list of dict
        Output from identify_emission_lines_with_pypeit.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    def gaussian(x, amp, mu, sigma, offset):
        return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + offset

    plt.figure(figsize=(12, 5))
    plt.plot(wave, cnts, 'k-', label='Spectrum')
    for line in lines:
        if line['fit_success']:
            xmod = np.linspace(line['center'] - 5*line['sigma'],
                               line['center'] + 5*line['sigma'], 100)
            plt.plot(xmod, gaussian(xmod, line['amplitude'], line['center'],
                                    line['sigma'], line['offset']),
                     '--', label=f"μ={line['center']:.1f}, σ={line['sigma']:.1f}")
    plt.xlabel('Wavelength')
    plt.ylabel('Counts')
    plt.title('Emission Line Identification and Gaussian Fits')
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir + 'emission_line_fits.png', dpi=300)
    plt.show()
    

def analyze_emission_lines_across_fibers(spec_sky, color='g', sigdetect=5.0, out_dir=None):
    """
    For each fiber in spec_sky[color]['all_spectra'], identify emission lines using
    identify_emission_lines_with_pypeit, then match lines across fibers and plot
    a 2D scatter of their fitted center vs. width for each matched line.
    Also plots the emission line fit for each fiber, offset vertically, using the same colors as in the scatter plot.

    Parameters
    ----------
    spec_sky : dict
        Output from extract_sky, must contain 'all_spectra' for the given color.
    color : str
        Channel to analyze ('r', 'g', or 'b').
    sigdetect : float
        Detection threshold for emission line finding.
    out_dir : str or None
        If provided, save the plot to this directory.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    wave = spec_sky[color]['wave']
    all_spectra = spec_sky[color]['all_spectra']
    n_fibers = all_spectra.shape[0]

    # Identify emission lines in each fiber
    all_lines = []
    for i in range(n_fibers):
        lines = identify_emission_lines_with_pypeit(wave, all_spectra[i], sigdetect=sigdetect)
        # Only keep lines with fit_success
        lines = [l for l in lines if l['fit_success']]
        all_lines.append(lines)

    # Gather all detected line centers for clustering
    all_centers = np.concatenate([[l['center'] for l in lines] for lines in all_lines if lines])

    if color == 'g':
        matched_lines = [(5567,5590), (6295, 6312), (6555, 6574)]
    else:

        # Cluster line centers using 1D binning (tolerance = 5 Angstrom)
        bins = np.arange(np.nanmin(all_centers)-1, np.nanmax(all_centers)+2, 5)
        hist, edges = np.histogram(all_centers, bins=bins)
        # Use bins with at least 2 detections as candidate lines
        matched_lines = []
        for i in range(len(hist)):
            if hist[i] >= 2:
                bin_min, bin_max = edges[i], edges[i+1]
                matched_lines.append((bin_min, bin_max))

    # Assign a color to each matched line group
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(matched_lines))))
    line_group_colors = {}
    for j, (bin_min, bin_max) in enumerate(matched_lines):
        line_group_colors[(bin_min, bin_max)] = colors[j]

    # For each matched line, collect (center, width) for each fiber where detected
    plt.figure(figsize=(8, 5))
    legend_entries = []
    for j, (bin_min, bin_max) in enumerate(matched_lines):
        centers = []
        widths = []
        for lines in all_lines:
            for l in lines:
                if bin_min <= l['center'] < bin_max:
                    centers.append(l['center'])
                    widths.append(l['sigma'])
        if len(centers) > 0:
            plt.scatter(centers, widths, s=40, color=colors[j], alpha=0.7, label=f"{np.mean(centers):.1f} Å")
            legend_entries.append(f"{np.mean(centers):.1f} Å")

    plt.xlabel('Emission Line Center (Å)')
    plt.ylabel('Line Width (σ, Å)')
    plt.title(f'Emission Line Fits Across Fibers ({color} channel)')
    if legend_entries:
        plt.legend(title='Line Group (mean center)', fontsize=8)
    plt.tight_layout()
    if out_dir is not None:
        plt.savefig(out_dir + f'emission_lines_{color}_fibers.png', dpi=300)
    plt.show()

    # Plot the emission line fit for each fiber, using the same colors as in the scatter plot
    def gaussian(x, amp, mu, sigma, offset):
        return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + offset

    plt.figure(figsize=(12, 2 + 0.5 * n_fibers))
    # Compute a minimal offset so that spectra do not overlap
    # Use the 95th percentile of the per-fiber (max-min) as a robust step
    per_fiber_range = [np.nanpercentile(s, 97.5) - np.nanpercentile(s, 2.5) for s in all_spectra]
    offset_step = np.nanmax(per_fiber_range) * 1.05 if per_fiber_range else 1.0
    for i in range(n_fibers):
        y_offset = i * offset_step
        plt.plot(wave, all_spectra[i] + y_offset, 'k-', lw=1)
        # Overplot each emission line fit in its group color
        for l in all_lines[i]:
            # Find which group this line belongs to
            group_color = None
            for (bin_min, bin_max), col in line_group_colors.items():
                if bin_min <= l['center'] < bin_max:
                    group_color = col
                    break
            if group_color is not None:
                xmod = np.linspace(l['center'] - 5*l['sigma'], l['center'] + 5*l['sigma'], 100)
                plt.plot(xmod, gaussian(xmod, l['amplitude'], l['center'], l['sigma'], l['offset']) + y_offset,
                         '--', color=group_color, lw=1.5)
        # Add fiber number text at the bottom left of this fiber's data
        min_x = np.nanmin(wave)
        min_y = np.nanmin(all_spectra[i]) + y_offset
        plt.text(min_x, min_y, f'Bench {spec_sky[color]["bench"][i]}, Fiber {spec_sky[color]["fiber"][i]}', color='black', fontsize=8, va='bottom', ha='left')

    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Counts (offset per fiber)')
    plt.title(f'Emission Line Fits for All Fibers ({color} channel)')
    plt.tight_layout()
    if out_dir is not None:
        plt.savefig(out_dir + f'emission_line_fits_{color}_fibers.png', dpi=300)
    plt.show()

def debug_extract_sky_fit(
    spec_sky, exobj, fibermap, arc, x_sci, y_sci, out_dir, line_center=5580.66, window=10, color='g'
):
    """
    Fit a Gaussian to the sky emission line at `line_center` in the specified channel,
    and plot a 2D histogram of fitted line center vs. width for all fibers.
    Also, plot the best-fit model for each fiber using the saved parameters.
    Only plot up to 6 fibers for the line fit plot.
    Assumes the Gaussian fit is always successful.
    """

    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    import numpy as np

    def gaussian(x, amp, mu, sigma, offset):
        return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + offset

    wave = spec_sky[color]['wave']
    all_spectra = spec_sky[color]['all_spectra']



    centers = []
    widths = []
    best_fit_params = []

    # Always compute map_data
    map_data = get_local_fiber_intensity_data(
        exobj, fibermap, arc, x_sci, y_sci, color, half_width=6.
    )

    # Fit each fiber's spectrum (assume fit always succeeds)
    for spectrum in all_spectra:
        mask = (wave > line_center - window) & (wave < line_center + window)
        x = wave[mask]
        y = spectrum[mask]
        amp_guess = np.nanmax(y) - np.nanmin(y)
        offset_guess = np.nanmedian(y)
        p0 = [amp_guess, line_center, 1.0, offset_guess]
        popt, _ = curve_fit(gaussian, x, y, p0=p0)
        centers.append(popt[1])
        widths.append(abs(popt[2]))
        best_fit_params.append(popt)

    # Plot line fits and best-fit models (max 6 fibers)
    import matplotlib.gridspec as gridspec

    # Find indices of 3 fibers with lowest centers and 3 with highest centers
    centers_arr = np.array(centers)
    if len(centers_arr) < 6:
        n_fibers = len(centers_arr)
        selected_indices = np.arange(n_fibers)
    else:
        lowest_indices = np.argsort(centers_arr)[:3]
        highest_indices = np.argsort(centers_arr)[-3:]
        selected_indices = np.concatenate([lowest_indices, highest_indices])
        n_fibers = len(selected_indices)
    fig = plt.figure(figsize=(14, 2.2 * n_fibers))
    gs = gridspec.GridSpec(n_fibers, 2, width_ratios=[3, 1], hspace=0.3, wspace=0.25)
    axs = []
    axs_map = []
    for i in range(n_fibers):
        axs.append(fig.add_subplot(gs[i, 0]))
        axs_map.append(fig.add_subplot(gs[i, 1]))

    for i, idx in enumerate(selected_indices):
        spectrum = all_spectra[idx]
        popt = best_fit_params[idx]
        mask = (wave > line_center - window) & (wave < line_center + window)
        x = wave[mask]
        y = spectrum[mask]
        xmod = np.linspace(line_center - window, line_center + window, 100)
        b = spec_sky[color]['bench'][idx]
        f = spec_sky[color]['fiber'][idx]
        axs[i].plot(x, y, 'k-', label=f'Bench {b}, Fiber {f}')
        label_fit = f'Best-fit (μ={popt[1]:.2f}, σ={abs(popt[2]):.2f})'
        axs[i].plot(xmod, gaussian(xmod, *popt), 'r--', label=label_fit)
        axs[i].set_ylabel('Counts')
        axs[i].legend(fontsize=7)

        # Plot local intensity map for this fiber
        if 'bench' in spec_sky[color] and 'fiber' in spec_sky[color]:
            b = spec_sky[color]['bench'][idx]
            f = spec_sky[color]['fiber'][idx]
            plot_local_fiber_intensity_on_axis(
                axs_map[i], map_data, x_sci, y_sci, color=color, half_width=6.
            )
            highlight_patch(axs_map[i], fibermap, b, f)
        else:
            axs_map[i].set_title('Map unavailable')
        axs_map[i].set_xticks([])
        axs_map[i].set_yticks([])
        axs_map[i].set_aspect('equal', adjustable='box')

    if n_fibers > 0:
        axs[-1].set_xlabel('Wavelength (Å)')
    plt.savefig(out_dir + 'sky_line_fits.png', dpi=300)
    plt.show()

    # Plot 2D scatter
    plt.figure(figsize=(6, 5))
    sc = plt.scatter(centers, widths, c=widths, cmap='viridis', s=40, edgecolor='k', alpha=0.8)
    # Highlight the selected fibers in the scatter plot
    plt.scatter(np.array(centers)[selected_indices], np.array(widths)[selected_indices], 
                c='red', s=80, edgecolor='black', marker='o', label='Selected fibers')
    plt.xlabel('Fitted Line Center (Å)')
    plt.ylabel('Fitted Line Width (σ, Å)')
    plt.title(f'Sky Line {line_center}Å Gaussian Fit ({color} Fibers)')
    plt.colorbar(label='Number of Fibers')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir + 'sky_line_2d_scatter.png', dpi=300)    
    plt.show()

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
    
    hstspec = np.loadtxt('/Users/emma/work/LLAMAS_UTILS/LLAMAS_UTILS/ffeige110.dat')
    
    rspec = np.loadtxt('/Users/emma/Desktop/plots/250214/F110_LLAMAS_2024-11-28T01_22_09.108_red_counts.txt')
    gspec = np.loadtxt('/Users/emma/Desktop/plots/250214/F110_LLAMAS_2024-11-28T01_22_09.108_green_counts.txt')
    uspec = np.loadtxt('/Users/emma/Desktop/plots/250214/F110_LLAMAS_2024-11-28T01_22_09.108_blue_counts.txt')
    
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
    
    fig, ax = plt.subplots(figsize=(20, 7), nrows=3, ncols=3)
    plt.suptitle('Sensitivity in red channel')
    
    ax[0][2].plot(rspec[:,0], rspec[:,1], '-r', label='Raw LLAMAS spectrum')
    ylim = ax[0][0].get_ylim()
    ax[1][2].plot(rspec[:,0], r_response, '-k', label='Sensitivity')
    # ax[0][2].set_ylim(ylim)
    ax[2][2].plot(rspec[:,0], hstmap(rspec[:,0]), '-k', label='HST Spectrum')
    ax[2][2].plot(rspec[:,0], r_response*rspec[:,1], '--r',
                  label='Flux-calibrated LLAMAS spectrum')
      
    ax[0][1].plot(gspec[:,0], gspec[:,1], '-g', label='Raw LLAMAS spectrum')
    ylim = ax[0][1].get_ylim()
    ax[1][1].plot(gspec[:,0], g_response, '-k', label='Sensitivity')
    # ax[0][1].set_ylim(ylim)
    ax[2][1].plot(gspec[:,0], hstmap(gspec[:,0]), '-k', label='HST Spectrum')
    ax[2][1].plot(gspec[:,0], g_response*gspec[:,1], '--g',
                  label='Flux-calibrated LLAMAS spectrum')

    ax[0][0].plot(uspec[:,0], uspec[:,1], '-b', label='Raw LLAMAS spectrum')
    ylim = ax[0][2].get_ylim()
    ax[1][0].plot(uspec[:,0], u_response, '-k', label='scaled Sensitivity')
    # ax[0][0].set_ylim(ylim)
    ax[2][0].plot(uspec[:,0], hstmap(uspec[:,0]), '-k', label='HST Spectrum')
    ax[2][0].plot(uspec[:,0], u_response*uspec[:,1], '--b', 
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
    import numpy as np
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

def plot_interp_vs_raw_fiber(spec_sky, exobj, arc, color, bench, fiber, out_dir):
    """
    Plot raw vs interpolated counts for a given fiber and bench.
    """
    import matplotlib.pyplot as plt

    # Get interpolated counts from spec_sky
    mask = (spec_sky[color]['bench'] == bench) & (spec_sky[color]['fiber'] == fiber)
    interp_cnts = spec_sky[color]['all_spectra'][mask][0]
    interp_wave = spec_sky[color]['wave']

    # Get raw counts and wavelength
    raw_wave, raw_cnts = extract_fiber_by_fibnum(exobj, arc, color, bench, fiber)

    # Plot
    plt.figure()
    plt.plot(raw_wave, raw_cnts, label='Raw Counts')
    plt.plot(interp_wave, interp_cnts, '--', label='Interpolated Counts')
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Counts')
    plt.legend()
    plt.title(f'Fiber {bench}-{fiber}: Raw vs Interpolated')
    plt.savefig(out_dir +  f'interp_vs_raw_{bench}_{fiber}.png', dpi=300)
    plt.show()

def refine_wavelength_with_telluric(waves, cnts, window=5):
    """
    Returns a refined wavelength grid and information for plotting the best-fit polynomial.
    """

    telluric_lines = [5580.7, 6302.9, 6565.7]

    lines = identify_emission_lines_with_pypeit(
        waves, cnts, sigdetect=5.0)
    # Match detected lines to ground truth telluric lines within the window
    detected_centers = np.array([l['center'] for l in lines if l['fit_success']])
    matched_gt = []
    matched_obs = []

    for gt_line in telluric_lines:
        # Find detected line closest to gt_line within window
        diffs = np.abs(detected_centers - gt_line)
        if np.any(diffs < window):
            idx = np.argmin(diffs)
            matched_gt.append(gt_line)
            matched_obs.append(detected_centers[idx])

    if len(matched_gt) < 2:
        # Not enough lines to fit a polynomial, return original grid and empty fit info
        return waves, None

    matched_gt = np.array(matched_gt)
    matched_obs = np.array(matched_obs)

    # Fit a 1st order polynomial (linear) mapping from observed to true
    coeffs = np.polyfit(matched_obs, matched_gt, deg=1)
    poly = np.poly1d(coeffs)
    refined_waves = poly(waves)

    # Return refined grid and fit info for plotting
    fit_info = {
        'matched_obs': matched_obs,
        'matched_gt': matched_gt,
        'coeffs': coeffs,
        'poly': poly
    }
    return refined_waves, fit_info

def plot_wavelength_refinement_fit(fit_info, out_dir=None):
    """
    Plot the polynomial fit and matched data points from refine_wavelength_with_telluric.

    Parameters
    ----------
    fit_info : dict
        Output from refine_wavelength_with_telluric (the 'fit_info' dictionary).
    out_dir : str or None
        If provided, save the plot to this directory.
    """
    import matplotlib.pyplot as plt

    matched_obs = fit_info['matched_obs']
    matched_gt = fit_info['matched_gt']
    poly = fit_info['poly']

    xfit = np.linspace(np.min(matched_obs) - 5, np.max(matched_obs) + 5, 100)
    yfit = poly(xfit)

    plt.figure(figsize=(7, 5))
    plt.plot(xfit, yfit, 'b-', label='Polynomial fit')
    plt.plot(matched_obs, matched_gt, 'ro', label='Matched points')
    for xo, yo in zip(matched_obs, matched_gt):
        plt.text(xo, yo, f"{yo:.1f}", color='red', fontsize=9, ha='right', va='bottom')
    plt.xlabel('Observed Line Center (Å)')
    plt.ylabel('True Telluric Line (Å)')
    plt.title('Wavelength Refinement Polynomial Fit')
    plt.legend()
    plt.tight_layout()
    if out_dir is not None:
        plt.savefig(out_dir + 'wavelength_refinement_fit.png', dpi=300)
    plt.show()

def extract_bjd_from_filename(filepath, ra_deg=153.42697, dec_deg=-45.28243, observatory_name='Las Campanas Observatory'):
    """
    Extracts the timestamp from a LLAMAS filename and converts it to BJD (TDB).
    
    Parameters
    ----------
    filepath : str
        Full path to the file (e.g., .../LLAMAS_2024-11-30T08_22_09.466_mef.fits)
    ra_deg : float
        Right ascension of the target in degrees.
    dec_deg : float
        Declination of the target in degrees.
    observatory_name : str
        Name of the observatory for astropy EarthLocation.
    
    Returns
    -------
    bjd : float
        Barycentric Julian Date (TDB) for the observation.
    """

    from astropy.time import Time
    from astropy.coordinates import EarthLocation, SkyCoord
    import os
    import re    
    # Extract timestamp from filename
    filename = os.path.basename(filepath)
    match = re.search(r'(\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2}\.\d+)', filename)
    if not match:
        raise ValueError("No timestamp found in filename.")
    timestamp = match.group(1).replace('_', ':')

    # Convert to astropy Time and compute BJD_TDB
    t = Time(timestamp, format='isot', scale='utc')
    t = t.tdb
    observatory = EarthLocation.of_site(observatory_name)
    target = SkyCoord(ra_deg, dec_deg, unit="deg")
    delta = t.light_travel_time(target, kind='barycentric', location=observatory)
    t = t + delta
    return t.mjd

def extract_all_spectra_and_bjd(filepaths, fibermap, arc, x, y, ra_deg=153.42697, dec_deg=-45.28243, observatory_name='Las Campanas Observatory'):
    """
    For a list of _mef.fits filepaths, extract sky-subtracted spectra and BJD.

    Returns:
        skysub_spectra_list: list of dicts (one per file, keys: 'r', 'g', 'b')
        bjd_list: list of BJD values
    """
    import pickle
    import os
    from astropy.time import Time
    from astropy.coordinates import EarthLocation, SkyCoord
    import re
    from llamas_pyjamas.config import OUTPUT_DIR

    skysub_spectra_list = []
    bjd_list = []

    for filepath in filepaths:
        ext = os.path.basename(filepath)[:-8]
        extract_pickle = os.path.join(OUTPUT_DIR, ext + 'extract.pkl')

        with open(extract_pickle, 'rb') as f:
            exobj = pickle.load(f)

        # Extract spectra
        spec = extract_aper(exobj, fibermap, arc, x, y)
        spec_sky = extract_sky(exobj, fibermap, arc, x, y, spec)
        spec_skysub = sky_subtraction(spec, spec_sky)
        skysub_spectra_list.append(spec_skysub)

        # Extract BJD from filename
        match = re.search(r'(\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2}\.\d+)', ext)
        if not match:
            raise ValueError(f"No timestamp found in filename: {filepath}")
        timestamp = match.group(1).replace('_', ':')
        t = Time(timestamp, format='isot', scale='utc')
        t = t.tdb
        observatory = EarthLocation.of_site(observatory_name)
        target = SkyCoord(ra_deg, dec_deg, unit="deg")
        delta = t.light_travel_time(target, kind='barycentric', location=observatory)
        t = t + delta
        bjd_list.append(t.mjd)

        # Apply heliocentric correction to each color channel
        for color in ['r', 'g', 'b']:
            wave = spec_skysub[color]['wave']
            # Use the observation time in ISO format
            obs_time = t.isot
            corrected_wave, v_helio_kms = apply_heliocentric_correction(
            wave, ra_deg, dec_deg, obs_time, observatory=observatory_name
            )
            spec_skysub[color]['wave'] = corrected_wave
            spec_skysub[color]['v_helio_kms'] = v_helio_kms

    return skysub_spectra_list, np.array(bjd_list)

def plot_skysub_spectra_by_color(skysub_spectra_list, bjd_list, out_dir=None, ext=''):
    """
    Plot sky-subtracted spectra for each color channel, with each spectrum labeled by BJD.
    Each figure corresponds to one color channel, arranged in at most 5 rows and as many columns as needed.

    Parameters
    ----------
    skysub_spectra_list : list of dicts
        Output from extract_skysub_spectra_and_bjd (list of spec_skysub dicts).
    bjd_list : array-like
        List or array of BJD values, same order as skysub_spectra_list.
    out_dir : str or None
        If provided, save the plot to this directory.
    ext : str
        Optional string to append to the filename.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import math

    colors = ['r', 'g', 'b']
    color_names = {'r': 'Red', 'g': 'Green', 'b': 'Blue'}

    for color in colors:
        n_spec = len(skysub_spectra_list)
        nrows = min(5, n_spec)
        ncols = math.ceil(n_spec / nrows)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 2.2 * nrows), sharex=True)
        axes = axes.flatten() if n_spec > 1 else [axes]
        for i, (spec, bjd) in enumerate(zip(skysub_spectra_list, bjd_list)):
            wave = spec[color]['wave']
            flux = spec[color]['spectrum']

            if 'mask' in spec[color]:
                mask = spec[color]['mask']
                wave = wave[mask]
                flux = flux[mask]            

            # Bin to 512 bins before plotting
            # winned, flux_binned = bin_spectrum(wave, flux, 512)
            axes[i].plot(wave, flux, label=f'BJD={bjd:.5f}', color=color)
            axes[i].legend(fontsize=8, loc='best')
            axes[i].set_ylabel('Counts')
        axes[0].set_title(f'{color_names[color]} Channel')
        axes[-1].set_xlabel('Wavelength (Å)')
        # Hide unused axes
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        plt.tight_layout()
        if out_dir is not None:
            plt.savefig(f"{out_dir}{ext}skysub_spectra_{color}.png", dpi=300)
        plt.show()

def plot_trailed_spectrum_from_skysub(
    skysub_spectra_list, bjd_list, color, line_center, period, t0, window=20, out_path=None, label=None
):
    """
    Plot a trailed spectrum for a given color channel using sky-subtracted spectra.

    Parameters
    ----------
    skysub_spectra_list : list of dicts
        Output from extract_skysub_spectra_and_bjd (list of spec_skysub dicts).
    bjd_list : array-like
        List or array of BJD values, same order as skysub_spectra_list.
    color : str
        Color channel to use ('r', 'g', or 'b').
    line_center : float
        Central wavelength (Angstroms) for the line.
    period : float
        Orbital period (days).
    t0 : float
        Reference time (days).
    window : float
        Half-width of the window (Angstroms) to cut around line_center.
    out_path : str or None
        If given, save the plot to this path.
    label : str or None
        Optional label for the plot title.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import interp1d

    # Gather wavelength and flux arrays for the chosen color
    wave_arr = [spec[color]['wave'] for spec in skysub_spectra_list]
    flux_arr = [spec[color]['spectrum'] for spec in skysub_spectra_list]
    BJD = np.array(bjd_list)

    phi = ((BJD - t0) / period) % 1

    # Interpolate all spectra onto a common wavelength grid centered on line_center
    # wave_template = wave_arr[0]
    # mask = (wave_template > line_center - window) & (wave_template < line_center + window)
    # wave_cut = wave_template[mask]

    wave_cut = np.linspace(line_center - window, line_center + window, 20)


    flux_interp = []
    for w, f in zip(wave_arr, flux_arr):
        interp_flux = interp1d(w, f, kind='linear', fill_value=np.nan, bounds_error=False)(wave_cut)
        flux_interp.append(interp_flux)
    flux_interp = np.array(flux_interp)

    # Compute radial velocity axis
    c = 2.99792458e5
    rv = ((wave_cut - line_center) / line_center) * c

    # Extend phase and flux for wraparound plotting
    phi_extended = np.concatenate([phi - 1, phi])
    flux_extended = np.concatenate([flux_interp, flux_interp], axis=0)

    plt.figure(figsize=(6, 3))
    plt.imshow(flux_extended.T, aspect='auto', cmap='inferno',
               extent=[phi_extended.min(), phi_extended.max(), rv.min(), rv.max()],
               origin='lower', interpolation='none')
    plt.colorbar(label='Flux')
    plt.xlabel('Orbital Phase')
    plt.ylabel('Radial Velocity (km/s)')
    title = f'Trailed Spectrum {label or f"{line_center:.1f} Å"}'
    plt.title(title)
    plt.xticks(ticks=np.linspace(-1, 1, 9))
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=300)
    plt.show()

def plot_line_window_from_skysub(
    skysub_spectra_list, bjd_list, color, line_center, t0, period, window=75, out_path=None, label=None
):
    """
    Plot a window around a provided line for all sky-subtracted spectra in a given color channel,
    arranging the plots in at most 5 rows and as many columns as needed.

    Parameters
    ----------
    skysub_spectra_list : list of dicts
        Output from extract_skysub_spectra_and_bjd (list of spec_skysub dicts).
    bjd_list : array-like
        List or array of BJD values, same order as skysub_spectra_list.
    color : str
        Color channel to use ('r', 'g', or 'b').
    line_center : float
        Central wavelength (Angstroms) for the line (e.g., 4686 for He II).
    t0 : float
        Reference time (days).
    period : float
        Orbital period (days).
    window : float
        Half-width of the window (Angstroms) to cut around line_center.
    out_path : str or None
        If given, save the plot to this path.
    label : str or None
        Optional label for the plot title.
    """
    import matplotlib.pyplot as plt
    import math
    import numpy as np

    n_spec = len(skysub_spectra_list)
    nrows = min(5, n_spec)
    ncols = math.ceil(n_spec / nrows)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3 * ncols, 2.2 * nrows), sharex=True)
    axes = axes.flatten() if n_spec > 1 else [axes]

    for i, (spec, bjd) in enumerate(zip(skysub_spectra_list, bjd_list)):
        wave = spec[color]['wave']
        flux = spec[color]['spectrum']
        mask = (wave > line_center - window) & (wave < line_center + window)
        phase = ((bjd - t0) / period) % 1
        axes[i].plot(wave[mask], flux[mask], label=f'BJD={bjd:.5f}, ϕ={phase:.2f}')
        axes[i].axvline(line_center, color='r', linestyle='--', alpha=0.5, label='Line Center' if i == 0 else None)
        axes[i].legend(fontsize=8, loc='best')
        axes[i].set_ylabel('Counts')
    axes[0].set_title(f'{label or f"Line {line_center:.1f} Å"} ({color} channel)')
    axes[-1].set_xlabel('Wavelength (Å)')

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=300)
    plt.show()

def bin_spectrum(wave, counts, nbins):
    """
    Bin the spectrum from its original wavelength bins to nbins.
    Returns the binned wavelength (center of bins) and binned counts (sum in each bin).
    """

    # Ensure arrays are numpy arrays
    wave = np.asarray(wave)
    counts = np.asarray(counts)

    # Define bin edges
    bin_edges = np.linspace(wave.min(), wave.max(), nbins + 1)
    # Bin centers
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Digitize wave to bins
    inds = np.digitize(wave, bin_edges) - 1
    # Clip indices to valid range
    inds = np.clip(inds, 0, nbins - 1)

    # Sum counts in each bin
    binned_counts = np.zeros(nbins)
    for i in range(nbins):
        mask = inds == i
        if np.any(mask):
            binned_counts[i] = np.nansum(counts[mask])
        else:
            binned_counts[i] = np.nan

    return bin_centers, binned_counts

def coadd_skysub_spectra(spec_skysub_list, color='g'):    
    """
    Coadd a list of sky-subtracted spectra (spec_skysub dicts) for a given color channel.
    All spectra are interpolated onto the wavelength grid of the first spectrum.

    Parameters
    ----------
    spec_skysub_list : list of dict
        Each dict is a sky-subtracted spectrum (output from sky_subtraction).
    color : str
        Color channel to coadd ('r', 'g', or 'b').

    Returns
    -------
    wave : ndarray
        Common wavelength grid (from the first spectrum).
    coadd_flux : ndarray
        Coadded flux array.
    """

    import numpy as np
    from scipy.interpolate import interp1d    

    if not spec_skysub_list:
        raise ValueError("Input list is empty.")

    # Use the wavelength grid of the first spectrum as the reference
    wave_ref = spec_skysub_list[0][color]['wave']
    fluxes_interp = []

    for spec in spec_skysub_list:
        wave = spec[color]['wave']
        flux = spec[color]['spectrum']
        # Apply mask if present
        if 'mask' in spec[color]:
            mask = spec[color]['mask']
            wave = wave[mask]
            flux = flux[mask]
        interp_flux = interp1d(wave, flux, kind='linear', bounds_error=False, fill_value=np.nan)(wave_ref)
        fluxes_interp.append(interp_flux)

    fluxes_interp = np.array(fluxes_interp)
    # Coadd using nanmean to ignore missing values
    coadd_flux = np.nanmean(fluxes_interp, axis=0)

    return wave_ref, coadd_flux

def coadd_all_colors(spec_skysub_list):
    """
    Coadd a list of sky-subtracted spectra for all three color channels.

    Parameters
    ----------
    spec_skysub_list : list of dict
        Each dict is a sky-subtracted spectrum (output from sky_subtraction).

    Returns
    -------
    coadd_results : dict
        Dictionary with keys 'r', 'g', 'b', each containing a tuple (wave, coadd_flux).
    """
    coadd_results = {}
    for color in ['r', 'g', 'b']:
        wave, coadd_flux = coadd_skysub_spectra(spec_skysub_list, color=color)
        coadd_results[color] = (wave, coadd_flux)
    return coadd_results

def plot_coadd_spectra_by_color(coadd_results, out_dir=None, label=None):
    """
    Plot the coadded spectra for all three color channels.

    Parameters
    ----------
    coadd_results : dict
        Dictionary with keys 'r', 'g', 'b', each containing a tuple (wave, coadd_flux).
    out_dir : str or None
        If provided, save the plots to this directory.
    label : str or None
        Optional label for the plot title.
    """
    import matplotlib.pyplot as plt

    color_map = {'r': 'red', 'g': 'green', 'b': 'blue'}
    for color in ['r', 'g', 'b']:
        wave, coadd_flux = coadd_results[color]
        plt.figure(figsize=(10, 4))
        plt.plot(wave, coadd_flux, color=color_map[color], label=label or f'Coadded {color} spectrum')
        plt.xlabel('Wavelength (Å)')
        plt.ylabel('Flux')
        plt.title(label or f'Coadded Spectrum ({color} channel)')
        plt.legend()
        plt.tight_layout()
        if out_dir:
            plt.savefig(f"{out_dir}coadd_{color}.png", dpi=300)
        plt.show()

def apply_heliocentric_correction(wavelength, ra, dec, obs_time, observatory='paranal'):
    """
    Apply heliocentric velocity correction to observed wavelengths.

    Parameters
    ----------
    wavelength : array-like
        Observed wavelengths (in Angstroms).
    ra : float
        Right ascension in degrees.
    dec : float
        Declination in degrees.
    obs_time : str or astropy.time.Time
        Observation time (ISO string or astropy Time object).
    observatory : str
        Observatory name for astropy EarthLocation.

    Returns
    -------
    corrected_wavelength : ndarray
        Heliocentric-corrected wavelengths (in Angstroms).
    v_helio_kms : float
        Heliocentric velocity correction (in km/s).
    """
    import astropy.units as u
    from astropy.coordinates import SkyCoord, EarthLocation
    from astropy.time import Time
    from astropy.constants import c
    import numpy as np    

    if not isinstance(obs_time, Time):
        obs_time = Time(obs_time, format='isot', scale='utc')
    location = EarthLocation.of_site(observatory)
    target = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
    v_helio = target.radial_velocity_correction(obstime=obs_time, location=location)
    v_helio_kms = v_helio.to(u.km/u.s).value
    corrected_wavelength = np.asarray(wavelength) * (1 + v_helio_kms / c.to('km/s').value)
    return corrected_wavelength, v_helio_kms