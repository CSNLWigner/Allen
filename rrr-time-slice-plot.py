
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

# Define the timepoints
timepoints = np.arange(0, preproc['stimulus-duration'], preproc['step-size'])  # in seconds

# print(mean_BU[np.newaxis].T)

# Define the colors
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
top_down_color = colors[0]
bottom_up_color = colors[1]

# Create the figure
fig, ax = plt.subplots()
plots.rrr_time_slice(ax, results, predictor_time, timepoints,
                     (top_down_color, bottom_up_color))

# Save the figure
save_fig('rrr-time-slice')
