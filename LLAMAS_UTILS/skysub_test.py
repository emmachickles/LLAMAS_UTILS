import os
from llamas_pyjamas.config import OUTPUT_DIR
from llamas_pyjamas.Image.WhiteLight import WhiteLightHex
import matplotlib.pyplot as plt
plt.ion()

plot_path = '/Users/emma/Desktop/plots/250601/'

fits_file = 'LLAMAS_2024-11-30T08_22_09.466_extract.pkl'
fits_path = os.path.join(OUTPUT_DIR, fits_file)
wl = WhiteLightHex(fits_path)

import pdb
pdb.set_trace()