import numpy as np
from astropy.io import fits
import pdb
import matplotlib.pyplot as plt

def print_header(mydir):
    cols = ['HIERARCH TEL RA', 'HIERARCH TEL DEC']
    
    import os
    fnames = os.listdir(mydir)
    fnames = [f for f in fnames if f[-5:] == '.fits']
    for f in fnames:
        print(f)
        hdul = fits.open(mydir + f)
        for c in cols:
            print(c)
            print(hdul[0].header[c])

def stretch_img(img, lo=5, up=95):
    log_img = np.log10(img - np.min(img) + 1)
    log_min = np.percentile(log_img, lo) # Clip bottom lo% in log scale
    log_max = np.percentile(log_img, up)  # Clip top up% in log scale
    clipped_log_img = np.clip(log_img, log_min, log_max)
    return clipped_log_img

def quick_plot(img, lo=5, up=95, stretch=True):
    if img.dtype == np.dtype('bool'):
        img = img.astype('int')
    if stretch:
        img = stretch_img(img, lo, up)
    plt.figure()
    plt.imshow(img, origin='lower', aspect='auto')
    plt.colorbar()

def plot_extract(color, bench, side, fiber, ext, sci_dir, scif, output_dir,
                 trace_dir='/home/echickle/work/LLAMAS_UTILS/LLAMAS_UTILS/mastercalib/'):
    import cloudpickle
    
    hdul = fits.open(sci_dir+scif)
    hdu = hdul[ext]
    sciimg = hdu.data
    
    fig, ax = plt.subplots(figsize=(10,14), sharex=True, nrows=4)
    data = np.loadtxt('Extract/'+scif[:-8]+'{}_{}_{}_{}.txt'.format(color, bench, side, fiber))
    ax[0].plot(data[:,0], data[:,1], '-', color=color, lw=1)

    with open(trace_dir+"LLAMAS_master_{}_{}_{}_traces.pkl".format(color, str(bench), side), "rb") as f:
        traceobj = cloudpickle.load(f)
    fiberimg = traceobj.fiberimg
    profimg = traceobj.profimg
    bpmask = traceobj.bpmask

    yextent = np.nonzero( fiberimg == fiber )[0]
    ymin = np.max([0,np.min(yextent)-5])
    ymax = np.max(yextent)+5
    extent = (0,2048, ymin, ymax)
    img = stretch_img( sciimg[ymin:ymax,:] )
    ax[1].imshow(img, origin='lower', aspect='auto',
                 extent=extent)
    ax[2].imshow(fiberimg[ymin:ymax,:], origin='lower', aspect='auto',
                 extent=extent)
    ax[3].imshow(profimg[ymin:ymax,:], origin='lower', aspect='auto',
                 extent=extent)
    plt.savefig(output_dir+'extract_'+scif[:-8]+'{}_{}_{}_{}.png'.format(color,bench,side,fiber))
    plt.close()

def plot_spec_all(ext_dir, scif, output_dir):
    for fiber in range(1,300):
        for bench in [1,2,3,4]:
            for side in ['A', 'B']:
                plot_spec(ext_dir, scif, output_dir, bench, side, fiber)
                plt.close()

def plot_spec(ext_dir, scif, output_dir, bench, side, fiber):

    rband = [6900, 9750]
    gband = [4750, 6900]
    uband = [3500, 4750]    
    
    fig, ax = plt.subplots(figsize=(12,8),nrows=3)    
    for i, color in enumerate(['red', 'green', 'blue']):
        data = np.loadtxt(ext_dir+scif[:-8]+'{}_{}_{}_{}.txt'.format(color, bench, side, fiber))

        if len(data.shape) > 1:
            wave = data[:,0]

            if color == 'red':
                band = rband
            elif color == 'green':
                band = gband
            elif color == 'blue':
                band = uband
            wave *= (band[1]-band[0]) / (np.max(wave)-np.min(wave))
            wave += band[0]

            ax[i].plot(wave, data[:,1], '-', color=color, lw=1)
            ax[i].set_xlabel('Approx. wavelength [Angstrom]')

        # print(color)
        # print(np.sum(data[:,1]))
    plt.tight_layout()
    plt.savefig(output_dir+'spec_'+scif[:-8]+'{}_{}_{}.png'.format(bench, side, fiber))
    
    

def plot_whitelight(ext_dir, scif, output_dir,
                    fibermap='/home/echickle/work/llamas-pyjamas/llamas_pyjamas/Image/LLAMAS_FiberMap_revA.dat'):
    fibermap = np.genfromtxt(fibermap, delimiter='|', dtype=None, autostrip=True, usecols=(1,2,3,4,5,6)).astype('str')

    fig, ax = plt.subplots(figsize=(20,12), ncols=3)
    for i, color in enumerate(['red', 'green', 'blue']):
        ax[i].set_title(color)
        ax[i].set_xlabel('xpos')
        ax[i].set_ylabel('ypos')
    
        xpos = []
        ypos = []
        flux = []
        for fiber in range(1,300):
            for bench in [1,2,3,4]:
                for side in ['A', 'B']:
                    data = np.loadtxt(ext_dir+scif[:-8]+'{}_{}_{}_{}.txt'.format(color, bench, side, fiber))
                    if len(data.shape) > 1:
                        flux.append(np.nansum(data[:,1]))
                        ind = np.nonzero( (fibermap[:,0] == str(bench)+side) * (fibermap[:,1] == str(fiber)) )[0][0]
                        xpos.append(fibermap[ind,4])
                        ypos.append(fibermap[ind,5])

        xpos, ypos = np.array(xpos).astype('float'), np.array(ypos).astype('float')
        flux = np.array(flux)
        from scipy.interpolate import LinearNDInterpolator
        interpolator = LinearNDInterpolator(list(zip(xpos, ypos)), flux)
        xmax = np.ceil(np.max(xpos))
        ymax = np.ceil(np.max(ypos))
        xx = np.linspace(0, xmax, 100)
        yy = np.linspace(0, ymax, 100)
        x_grid, y_grid = np.meshgrid(xx, yy)
        whitelight = interpolator(x_grid, y_grid)

        
        ax[i].imshow(whitelight, origin='lower', aspect='auto',
                     extent=[0, xmax, 0, ymax])

    out = output_dir+'whitelight_'+scif[:-9]+'.png'
    plt.tight_layout()
    plt.savefig(out)
    print('Saved '+out)
    

    
    
