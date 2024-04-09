

import numpy as np
from utils.data_io import load_pickle
import matplotlib.pyplot as plt
import yaml

# Load rrr-param-search parameters
search = yaml.safe_load(open("params.yaml"))["rrr-param-search"]
rrr = yaml.safe_load(open("params.yaml"))["rrr"]

# Load cross-time-RRR from results
matrix = load_pickle("cross-time-RRR", path="results")

# The diagonal of the matrix should be nan
np.fill_diagonal(matrix, np.nan)

# timeseries = np.arange(0, preprocess["stimulus-duration"], search['lag']/1000)
timeseries = np.array(search['lag'])

# Add the first timepoint to each element in timeseries
timeseries = timeseries + search['timepoints'][0]

# Plot the matrix. colormap do not use white color. Make the resolution higher.
plt.imshow(matrix, cmap='terrain', interpolation='bilinear')
plt.xticks(range(timeseries.shape[0]), timeseries)
plt.yticks(range(timeseries.shape[0]), timeseries)
plt.xlabel(f"Timepoints of {rrr['predictor']}")
plt.ylabel(f"Timepoints of {rrr['target']}")
plt.colorbar()

# Save the plot
plt.savefig("figures/cross-time-RRR.png")
