
from matplotlib import pyplot as plt
import numpy as np
from utils.data_io import load_pickle, save_fig
import yaml

# Load parameters
preproc = yaml.safe_load(open('params.yaml'))['preprocess']
predictor_time = yaml.safe_load(open('params.yaml'))['rrr-time-slice']['predictor-time']

# Load results
results = load_pickle('rrr-time-slice')
mean_TD = results['top-down']['mean']
mean_BU = results['bottom-up']['mean']

# Define the timepoints
timepoints = np.arange(0, preproc['stimulus-duration'], preproc['step-size'])  # in seconds

# print(mean_BU[np.newaxis].T)

# Plot the results
plt.plot(timepoints, mean_TD, label='Top-down')
plt.plot(timepoints, mean_BU, label='Bottom-up')
plt.xlabel('Time (s)')
plt.ylabel('R^2')
# Make a vertical line at the predictor time
plt.axvline(x=predictor_time, color='k', linestyle='--')
plt.xticks([0, predictor_time, timepoints[-1]])
plt.xlim([0, timepoints[-1]])
plt.legend()

# Save the figure
plt.savefig('figures/rrr-time-slice.png')
