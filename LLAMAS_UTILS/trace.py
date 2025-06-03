import sys
sys.path.append('/Users/emma/projects/llamas-pyjamas')  
import os
from llamas_pyjamas.Trace.traceLlamas import TraceLlamas
import cloudpickle
import matplotlib.pyplot as plt

import numpy as np


out_dir = '/Users/emma/Desktop/work/250417/'

with open("/Users/emma/projects/llamas-pyjamas/llamas_pyjamas/mastercalib/LLAMAS_master_green_4_A_traces.pkl", "rb") as f:
    traceobj = cloudpickle.load(f)
print(dir(traceobj))
print(traceobj.__dict__.keys())



plt.ion()
plt.figure()
plt.title('Lists the fiber # of each pixel')
plt.imshow(traceobj.fiberimg, origin='lower', aspect='auto')

plt.figure()
plt.title('bad pixel mask')
plt.imshow(traceobj.bpmask, origin='lower', aspect='auto')

plt.figure()
plt.title('Profile weighting function')
plt.imshow(traceobj.profimg, origin='lower', aspect='auto')
for i in range(len(traceobj.traces)):
    plt.plot(traceobj.traces[i], '-r', lw=1)

# traceobj.traces

filepath = '/Users/emma/projects/llamas-data/standard/LLAMAS_2024-11-28T01_22_09.108_mef.fits'
from llamas_pyjamas.File.llamasIO import process_fits_by_color
hdu = process_fits_by_color(filepath)

hduidx = 20
img = hdu[hduidx].data
# img = np.fliplr(img)
# img = np.flipud(img)

fiber = 257
mask = ( (traceobj.fiberimg >= 253) * (traceobj.fiberimg <= 257) )
rows = np.any(mask, axis=1)
row_inds = np.where(rows)[0]
row_min = row_inds.min()
row_max = row_inds.max()

# img = img[row_min:row_max + 1, :]

plt.figure()
plt.title('Bench {} Side {} Color {}'.format(hdu[hduidx].header['BENCH'], hdu[hduidx].header['SIDE'], hdu[hduidx].header['COLOR']))
plt.imshow(img, origin='lower', aspect='auto')
for i in [253, 255, 257]: # range(len(traceobj.traces)):
    plt.plot(traceobj.traces[i], '-r', lw=1)
plt.ylim(row_min, row_max)
plt.colorbar()
plt.savefig(out_dir+'LLAMAS_2024-11-28T01_22_09.108_mef_green_traces.png', dpi=300)

# hduidx = 23
# plt.figure()
# plt.title('Bench {} Side {} Color {}'.format(hdu[hduidx].header['BENCH'], hdu[hduidx].header['SIDE'], hdu[hduidx].header['COLOR']))
# plt.imshow(hdu[hduidx].data, origin='lower', aspect='auto')
# plt.colorbar()