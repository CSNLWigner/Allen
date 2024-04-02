
from matplotlib import pyplot as plt
import numpy as np
from utils.data_io import load_pickle, save_fig
import yaml

# Load parameters
preproc = yaml.safe_load(open('params.yaml'))['preprocess']
search_params = yaml.safe_load(open('params.yaml'))['rrr-param-search']


# Load lags
lags = load_pickle('max-lags-along-time')

# Define the timepoints
timepoints = np.arange(0, len(lags)*preproc['step-size'], preproc['step-size']) # in seconds

# Plot the max ranks
plt.plot(timepoints, lags)
plt.xlabel('Time (s)')
plt.ylabel('Max lag')
plt.xticks(timepoints[::5], timepoints[::5])

# Save the figure
plt.savefig('figures/max-lags-along-time.png')
