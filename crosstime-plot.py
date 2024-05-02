

import numpy as np
from utils.data_io import load_pickle
import matplotlib.pyplot as plt
import yaml

# Load rrr-param-search parameters
search = yaml.safe_load(open("params.yaml"))["rrr-param-search"]
rrr = yaml.safe_load(open("params.yaml"))["rrr"]

# Load cross-time-RRR from results
matrix = load_pickle("cross-time-RRR", path="results")

# print(matrix)

# The diagonal of the matrix should be nan
np.fill_diagonal(matrix, np.nan)

# timeseries = np.arange(0, preprocess["stimulus-duration"], search['lag']/1000)
timeseries = np.arange(0, 200, 3)
# timeseries = np.array(search['lag'])

# Add the first timepoint to each element in timeseries
timeseries = timeseries + search['timepoints'][0]

# tick frequency
tick_frequency = 5

# Plot the matrix. colormap do not use white color. Make the resolution higher.
plt.imshow(matrix, cmap='terrain', interpolation='bilinear')
plt.xticks(range(0, timeseries.shape[0], tick_frequency), timeseries[::tick_frequency])
plt.yticks(range(0, timeseries.shape[0], tick_frequency), timeseries[::tick_frequency])
plt.xlabel(f"Timepoints of {rrr['target']}")
plt.ylabel(f"Timepoints of {rrr['predictor']}")
plt.colorbar()

# Save the plot
plt.savefig("figures/cross-time-RRR.png")
