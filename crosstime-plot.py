

import numpy as np
from utils.data_io import load_pickle, save_fig
from utils import plots
import matplotlib.pyplot as plt
import yaml
import sys

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
