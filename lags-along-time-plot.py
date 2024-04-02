
from matplotlib import pyplot as plt
import numpy as np
from utils.data_io import load_pickle, save_fig
import yaml

# Load parameters
preproc = yaml.safe_load(open('params.yaml'))['preprocess']
search_params = yaml.safe_load(open('params.yaml'))['rrr-param-search']


# Load lags
lags = load_pickle('lags-along-time')

# Define the axes ticks
lag_times = search_params['lag'][:len(lags)]
timepoints = np.arange(0, preproc['stimulus-duration'], preproc['step-size']) # in seconds

# Plot the max ranks
plt.imshow(lags.T, aspect='auto', origin='lower', extent=[0, preproc['stimulus-duration'], 0, search_params['lag'][-1]])
plt.xlabel('Time (s)')
plt.ylabel('Lag (ms)')
# plt.xticks(np.arange(0, preproc['stimulus-duration'], 5), np.arange(0, preproc['stimulus-duration'], 5))
plt.yticks(np.arange(0, search_params['lag'][-1], 5), np.arange(0, search_params['lag'][-1], 5))

plt.colorbar(label='R^2')

# Save the figure
plt.savefig('figures/lags-along-time.png')