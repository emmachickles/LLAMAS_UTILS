import numpy as np
from astropy.io import fits
import pdb
import matplotlib.pyplot as plt

def extract_trace(sciimg, fiberimg, profimg, bpmask, fiber):
    x, y = [], []
    for c in range(2048): # number of columns
        good_inds = np.nonzero( (fiberimg[:,c] == fiber) * ~bpmask[:,c])[0]
        weights = profimg[good_inds,c]

        if len(good_inds) > 0:
            x.append(c)
            y.append( np.sum(sciimg[good_inds,c] * weights) )
            
    x = np.array(x)
    y = np.array(y)
            
    return x, y

def extract_ccd(scif, sci_dir, ext_dir, mastercalib='/home/echickle/work/LLAMAS_UTILS/LLAMAS_UTILS/mastercalib/'):

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

            outf = ext_dir+scif[:-8]+'{}_{}_{}_{}.txt'.format(color, bench, side, fiber)
            np.savetxt(outf, np.array([x,y]).T)
            
            print('Saved '+outf)
            
def get_spec(x, y, fibermap='/home/echickle/work/llamas-pyjamas/llamas_pyjamas/Image/LLAMAS_FiberMap_revA.dat'):
    fibermap = np.genfromtxt(fibermap, delimiter='|', dtype=None, autostrip=True, usecols=(1,2,3,4,5,6)).astype('str')
    xpos = fibermap[1:,4].astype('float')
    ypos = fibermap[1:,5].astype('float')

    dist = (xpos-x)**2 + (ypos-y)**2

    inds = np.argsort(dist)

    return fibermap[1:][inds]
