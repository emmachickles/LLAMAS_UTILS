from llamas_pyjamas.File.llamasIO import process_fits_by_color
from llamas_pyjamas.GUI.guiExtract import match_hdu_to_traces
from llamas_pyjamas.QA.llamasQA import plot_traces_on_image
from llamas_pyjamas.config import CALIB_DIR
import glob, os, cloudpickle
import matplotlib.pyplot as plt
import numpy as np

filepath = '/Users/emma/projects/llamas-data/ATLASJ1138/LLAMAS_2024-11-28T07_41_00.294_mef.fits'
out_dir = '/Users/emma/Desktop/work/2505013/'

hdu = process_fits_by_color(filepath)

masterfile = 'LLAMAS_master'
trace_files = glob.glob(os.path.join(CALIB_DIR, f'{masterfile}*traces.pkl'))

hdu_trace_pairs = match_hdu_to_traces(hdu, trace_files)

for hdu_index, trace_file in hdu_trace_pairs[-6:]:
    hdu_data = hdu[hdu_index].data
    with open(trace_file, mode='rb') as f:
        traceobj = cloudpickle.load(f)
    plot_traces_on_image(traceobj, hdu_data, zscale=True)


