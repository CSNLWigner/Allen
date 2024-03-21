
from matplotlib import pyplot as plt
import numpy as np
from utils.data_io import load_pickle, save_fig
import yaml

# Load parameters
search_params = yaml.safe_load(open('params.yaml'))['rrr-param-search']

# Define the timepoints
lag_times = search_params['lag']

# Load max ranks
lags = load_pickle('time-lag-search')

# Plot the max ranks
plt.plot(lag_times, lags)
plt.xlabel('Lag (s)')
plt.ylabel('R^2')
plt.xticks(lag_times[::2], lag_times[::2])

# Save the figure
plt.savefig('figures/time-lag-search.png')
