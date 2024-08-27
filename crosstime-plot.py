# crosstime-plot.py

"""
This module plots the cross-time RRR analysis.

**Parameters**:

- `load`:
    - `session`: The session to analyze.
- `rrr-param-search`: The RRR parameter search parameters.
- `rrr`: The RRR parameters.
- `crosstime`: The cross-time parameters.

**Input**:

- `results/cross-time-RRR.pickle`: Pickle file containing the cross-time RRR analysis results.

**Output**:

- `figures/cross-time-RRR.png`: The cross-time RRR plot.

(Also saves a copy of the plot in the `cache` directory with the session name appended to the filename.)

**Submodules**:

- `utils.data_io`: Module for loading and saving data.
- `utils.plots`: Module for plotting functions.

"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import yaml

from utils import plots
from utils.data_io import load_pickle, save_fig

# Get the parameters from the command line
prediction_direction = sys.argv[1]

# Load rrr-param-search parameters
session = yaml.safe_load(open("params.yaml"))["load"]["session"]
search = yaml.safe_load(open("params.yaml"))["rrr-param-search"]
rrr = yaml.safe_load(open("params.yaml"))["rrr"]
params = yaml.safe_load(open("params.yaml"))["crosstime"]

# Load cross-time-RRR from results
matrix = load_pickle("cross-time-RRR", path="results")

# print(matrix)

# Define the parameters
scaling_factor = params["scaling-factor"]
timeseries = np.arange(0, 200, scaling_factor)

# Add the first timepoint to each element in timeseries
timeseries = timeseries + search['timepoints'][0]

# Create a plot with 2 columns
fig, ax = plt.subplots()
plots.crosstime_RRR(ax, matrix, rrr['predictor'], rrr['target'], timeseries)

# Save the plot
save_fig(fig, "cross-time-RRR")
save_fig(fig, f"{prediction_direction}_cross-time-RRR_{session}", path="cache")
