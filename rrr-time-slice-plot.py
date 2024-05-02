
from matplotlib import pyplot as plt
import numpy as np
from utils.data_io import load_pickle, save_fig
from utils import plots
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

# Create the figure
fig, ax = plt.subplots()
plots.rrr_time_slice(ax, mean_TD, mean_BU)

# Save the figure
save_fig('rrr-time-slice')
