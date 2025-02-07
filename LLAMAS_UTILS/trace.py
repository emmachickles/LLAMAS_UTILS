import sys
sys.path.append('/home/echickle/work/llamas-pyjamas')  
from llamas_pyjamas.Trace.traceLlamas import TraceLlamas
import cloudpickle

with open("mastercalib/LLAMAS_master_blue_4_A_traces.pkl", "rb") as f:
    traceobj = cloudpickle.load(f)
print(dir(traceobj))
print(traceobj.__dict__.keys())

import matplotlib.pyplot as plt
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
