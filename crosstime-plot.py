

import numpy as np
from utils.data_io import load_pickle
import matplotlib.pyplot as plt
import yaml

# Load rrr-param-search parameters
search = yaml.safe_load(open("params.yaml"))["rrr-param-search"]

# Load cross-time-RRR from results
matrix = load_pickle("cross-time-RRR", path="results")

# timeseries = np.arange(0, preprocess["stimulus-duration"], search['lag']/1000)
timeseries = np.array(search['lag'])

# Add the first timepoint to each element in timeseries
timeseries = timeseries + search['timepoints'][0]

# Plot the matrix
plt.imshow(matrix, cmap='hot', interpolation='nearest')
plt.xticks(range(timeseries.shape[0]), timeseries)
plt.yticks(range(timeseries.shape[0]), timeseries)
plt.colorbar()

# Save the plot
plt.savefig("figures/cross-time-RRR.png")
