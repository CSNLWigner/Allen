

from matplotlib import pyplot as plt
import numpy as np
from utils.data_io import load_pickle, save_fig
from utils.plots import cv_rank_time_plot
import yaml

# Load the params
preproc = yaml.safe_load(open('params.yaml'))['preprocess']
params = yaml.safe_load(open('params.yaml'))['rrr-cv-rank-time']

# Define the cross-validation, and time
cv = params['cv']
time_lag = params['lag']
timepoints = params['timepoints']

# Load the results
result = load_pickle('CV-lag-time')

# Create fig, axs with 2 rows and 1 col
fig, axs = plt.subplots(2, 1, figsize=(10, 10))

# Plot the mean above the CVs
mean_cv = np.nanmean(result, axis=0)
print(mean_cv)
im = cv_rank_time_plot(mean_cv,
                        title=f'Averaged over CVs', ax=axs[0],
                        xlabel='Lag', ylabel='Time',
                        xticks=time_lag, yticks=timepoints)
fig.colorbar(im, ax=axs[0])

print('result.shape:', result.shape)

# Plot the mean above the times
mean_time = np.nanmean(result, axis=2)
print(mean_time)
im = cv_rank_time_plot(mean_time,
                        title=f'Averaged over times', ax=axs[1],
                        xlabel='CV', ylabel='Lag',
                        xticks=cv, yticks=time_lag)
fig.colorbar(im, ax=axs[1])


# Plot the results

# Save the figure
save_fig(fig, f'CV-timelag')