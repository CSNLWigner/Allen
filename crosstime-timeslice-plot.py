# crosstime-timeslice-plot.py

"""
This module plots the cross-time RRR analysis and time-slice analysis based on the cached results.

**Parameters**:

- `load`:
    - `session`: The session to analyze.
- `preprocess`: The preprocess parameters.
- `rrr-time-slice`: The time-slice parameters.

**Input**:

- `cache/top-down_cross-time-RRR_<session>.pickle`: Pickle file containing the top-down cross-time RRR analysis results.
- `cache/bottom-up_cross-time-RRR_<session>.pickle`: Pickle file containing the bottom-up cross-time RRR analysis results.
- `results/rrr-time-slice.pickle`: Pickle file containing the time-slice analysis results.

**Output**:

- `figures/crosstime-timeslice-plot.png`: The cross-time RRR and time-slice plot.

(Also saves a copy of the plot in the `cache` directory with the session name appended to the filename.)

**Submodules**:

- `utils.data_io`: Module for loading and saving data.
- `utils.plots`: Module for plotting functions.

"""

import numpy as np
import yaml
from matplotlib import pyplot as plt

from utils import plots
from utils.data_io import load_pickle, save_fig

# Load the params
load = yaml.safe_load(open('params.yaml'))['load']
preprocess = yaml.safe_load(open('params.yaml'))['preprocess']
timesliceparam = yaml.safe_load(open('params.yaml'))['rrr-time-slice']

# Define the variables
session = load['session']
scaling_factor = 3
step_size = preprocess['step-size']
stim_dur = preprocess['stimulus-duration']
timestep = stim_dur / step_size

# Load the results
crosstest_TD = load_pickle(f'top-down_cross-time-RRR_{session}', path='cache/crosstime')
crosstest_BU = load_pickle(f'bottom-up_cross-time-RRR_{session}', path='cache/crosstime')
timeslice = load_pickle('rrr-time-slice', path='results')

# Create the figure
fig, axs = plt.subplots(1, 3, figsize=(20, 5))

cax = plots.crosstime_RRR(axs[0], crosstest_TD, predictor='VISl', target='VISp',
                    timeseries=np.arange(0, 200, scaling_factor))
axs[0].set_title('Top-down')
fig.colorbar(cax, ax=axs[0])

cax = plots.crosstime_RRR(axs[1], crosstest_BU, predictor='VISp', target='VISl',
                    timeseries=np.arange(0, 200, scaling_factor))
axs[1].set_title('Bottom-up')
fig.colorbar(cax, ax=axs[1])

plots.rrr_time_slice(axs[2], timeslice['top-down']['mean'], timeslice['bottom-up']['mean'],
                     timepoints=np.arange(0, stim_dur, step_size), predictor_time=timesliceparam['predictor-time'])
axs[2].set_title('Time-slice')

# Save the figure
save_fig(fig, 'crosstime-timeslice-plot')
save_fig(fig, f'crosstime-timeslice-plot_{session}', path='cache')
