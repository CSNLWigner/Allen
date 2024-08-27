# rrr-time-slice-plot.py

"""
This module plots the results of the RRR time-slice analysis.

**Parameters**:

- `preprocess`: Preprocess parameters.
- `rrr-time-slice`: Time-slice parameters.
    - `predictor-time`: Time of the predictor stimulus.

**Input**:

- `results/rrr-time-slice.pickle`: Results of the time-slice analysis.

**Output**:

- `figures/rrr-time-slice.png`: Plot of the RRR time-slice analysis.

**Submodules**:

- `utils.plots`: Module for plotting functions.
- `utils.data_io`: Module for loading and saving data.

"""
import numpy as np
import yaml
from matplotlib import pyplot as plt

from utils import plots
from utils.data_io import load_pickle, save_fig

# Load parameters
preproc = yaml.safe_load(open('params.yaml'))['preprocess']
predictor_time = yaml.safe_load(open('params.yaml'))['rrr-time-slice']['predictor-time']

# Load results
results = load_pickle('rrr-time-slice')

# Define the timepoints
timepoints = np.arange(0, preproc['stimulus-duration'], preproc['step-size'])  # in seconds

# print(mean_BU[np.newaxis].T)

# Define the colors
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
top_down_color = colors[0]
bottom_up_color = colors[1]

# Create the figure
fig, ax = plt.subplots()
plots.rrr_time_slice(ax, results, predictor_time, timepoints,
                     (top_down_color, bottom_up_color))

# Save the figure
save_fig('rrr-time-slice')
