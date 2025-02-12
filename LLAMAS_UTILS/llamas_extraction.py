# Based on lalmas_pyjamas/Tutorials/llamas_extraction_demo.ipynb

filepath = '/Users/emma/projects/llamas-test/standard/LLAMAS_2024-11-28T01_22_09.108_mef.fits'

import os
import sys
import numpy as np
import pickle
import ray
import pkg_resources
import glob
import traceback
from   pathlib import Path
from   llamas_pyjamas.config import BASE_DIR, OUTPUT_DIR, DATA_DIR

# Get package root and add to path before other imports as a precaution -> if installed as package this should hopefully not be needed
# package_root = Path().absolute().parent
# sys.path.append(str(package_root))
sys.path.append(BASE_DIR+'/')


ray.init(ignore_reinit_error=True)
import llamas_pyjamas.Trace.traceLlamasMulti as trace # type: ignore
import llamas_pyjamas.Extract.extractLlamas as extract # type: ignore
from llamas_pyjamas.Image.WhiteLight import WhiteLight, WhiteLightFits

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

from llamas_pyjamas.GUI.guiExtract import GUI_extract
extract_pickle = package_path+'/output/'+filepath.split('/')[-1][:-8]+'extract.pkl'
if not os.path.exists(extract_pickle):
    GUI_extract(filepath)
    
with open(extract_pickle, 'rb') as f:
    exobj = pickle.load(f)