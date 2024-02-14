
# Load params
from matplotlib import pyplot as plt
import numpy as np
import yaml

from utils.data_io import load_pickle, save_fig, save_pickle
from utils.utils import get_time

# Load the parameters
preprocess = yaml.safe_load(open('params.yaml'))['preprocess']

# generate_values = lambda range_end, bin_size: [round(t*bin_size, 3) for t in range(range_end)]
# get_time = lambda time_bin, bin_size: round(time_bin*bin_size, 3)

# Load the data
V1_activity = load_pickle("5_block_VISp-activity", path="data/area-responses")

# Get the shape of the data
N, K, T = V1_activity.shape

# Calculate the frequency histogram of the data for each time bin
histograms = []
for t in range(T):
    # bins=preprocess['histogram-bins']
    histograms.append(np.histogram(V1_activity[:, :, t])[0])
# histograms = np.array(histograms)

# Save the histograms
save_pickle(histograms, "5_block_VISp-histograms", path="results")

# Plot the histograms
fig, ax = plt.subplots(1, len(histograms), figsize=(20, 5))
for t in range(len(histograms)):
    ax[t].bar(range(len(histograms[t])), histograms[t])
    ax[t].set_title(f'{get_time(t)}-{get_time(t+1)} s')
    ax[t].set_xlabel('Bin')
    ax[t].set_ylabel('Count')

# show the plot
plt.show()

# Save the figure
save_fig(fig, "V1-histograms_on_natural_images", path="figures")
