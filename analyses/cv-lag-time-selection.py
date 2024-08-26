# analyses/cv-lag-time-selection.py

"""
This module contains tools for selecting the maximum mean across the CV, lag, and time dimensions.

Functions:
- None
"""


# Calculate the mean across each dimensions separately
import numpy as np

from utils.data_io import load_pickle, save_pickle

# Load the results
result = load_pickle('CV-lag-time')

# Calculate the mean across each dimension
mean_cv = np.nanmean(result, axis=0)
mean_lag = np.nanmean(result, axis=1)
mean_time = np.nanmean(result, axis=2)

# Select the mean matrix that has the highest overall values
max_mean = np.argmax(
    [np.nanmean(mean_cv), np.nanmean(mean_lag), np.nanmean(mean_time)])
print(f"The maximum mean is across {['cv', 'lag', 'time'][max_mean]}.")
if max_mean == 0:
    selected_mean = mean_cv
elif max_mean == 1:
    selected_mean = mean_lag
else:
    selected_mean = mean_time

# Save the results
save_pickle(selected_mean, f'CV-lag-time_max-mean')
# Save the name of the selected mean
save_pickle(['cv', 'lag', 'time'][max_mean], f'CV-lag-time_max-mean_name')
