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

def extract_trace(sciimg, fiberimg, profimg, bpmask, fiber):
    x, y = [], []
    for c in range(2048): # number of columns
        good_inds = np.nonzero( (fiberimg[:,c] == fiber) * ~bpmask[:,c])[0]
        weights = profimg[good_inds,c]

        if len(good_inds) > 0:
            # plt.ion()
            # quick_plot(fiberimg)
            # quick_plot(fiberimg[:,c:c+10], stretch=False)
            # quick_plot(bpmask[:,c:c+10], stretch=False)
            # quick_plot(profimg[:,c:c+10])

            x.append(c)
            y.append( np.sum(sciimg[good_inds,c] * weights) )
    x = np.array(x)
    y = np.array(y)
            
    return x, y

def extract_ccd(scif, sci_dir, out_dir='Raw/', mastercalib='/home/echickle/work/LLAMAS_UTILS/LLAMAS_UTILS/mastercalib/'):

    from astropy.io import fits
    hdul = fits.open(sci_dir+scif)
    
    import sys
    sys.path.append('/home/echickle/work/llamas-pyjamas')  
    from llamas_pyjamas.Trace.traceLlamas import TraceLlamas
    import cloudpickle

    for ext in range(1, len(hdul)):
        hdu = hdul[ext]
        sciimg = hdu.data

        color = hdu.header['COLOR']
        bench = hdu.header['BENCH']
        side = hdu.header['SIDE']

        with open(mastercalib+"LLAMAS_master_{}_{}_{}_traces.pkl".format(color, bench, side), "rb") as f:
            traceobj = cloudpickle.load(f)

        fiberimg = traceobj.fiberimg
        profimg = traceobj.profimg
        bpmask = traceobj.bpmask

        for fiber in range(1, 300):

            x, y = extract_trace(sciimg, fiberimg, profimg, bpmask, fiber)

            outf = out_dir+scif[:-8]+'{}_{}_{}_{}.txt'.format(color, bench, side, fiber)
            np.savetxt(outf, np.array([x,y]).T)
            
            print('Saved '+outf)
            
            
