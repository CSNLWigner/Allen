

import numpy as np
from utils.data_io import load_pickle, save_fig
from utils import plots
import matplotlib.pyplot as plt
import yaml

# Load rrr-param-search parameters
search = yaml.safe_load(open("params.yaml"))["rrr-param-search"]
rrr = yaml.safe_load(open("params.yaml"))["rrr"]

# Load cross-time-RRR from results
matrix = load_pickle("cross-time-RRR", path="results")

# print(matrix)

# timeseries = np.arange(0, preprocess["stimulus-duration"], search['lag']/1000)
timeseries = np.arange(0, 200, 3)
# timeseries = np.array(search['lag'])

# Add the first timepoint to each element in timeseries
timeseries = timeseries + search['timepoints'][0]

# Create a plot with 2 columns
fig, ax = plt.subplots(1, 3, figsize=(12, 6))
plots.crosstime_RRR(ax, matrix, rrr['predictor'], rrr['target'], timeseries)

# Save the plot
save_fig("cross-time-RRR")
