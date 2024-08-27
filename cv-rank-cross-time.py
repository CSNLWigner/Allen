# cv-rank-cross-time.py

"""
This module searches for the optimal cv-fold and time lag in the RRR model using cross-validation.

The script loops through the cross-validation folds, time lags, and ranks specified in the parameters and calculates the RRR model for each combination. The results are saved as a pickle file in the `data/rrr-results` directory.

**Parameters**:

- `rrr-cv-rank-time`:
    - `cv`: A list of cross-validation folds to use.
    - `ranks`: A list of ranks to use.
    - `duration`: The duration of the time window.
    - `time-bin`: The time bin size.
    - `time-step`: The time step size.
    
**Input**:

- `data/area-responses/5_block_VISp-activity.pickle`: Pickle file containing the raw activity data for the VISp brain area.
- `data/area-responses/5_block_VISl-activity.pickle`: Pickle file containing the raw activity data for the VISl brain area.

**Output**:

- `data/rrr-results/CV-rank.pickle`: Pickle file containing the results of the cross-validation of the rank in the RRR model. Shape: (n_cv, n_rank)
- `figures/CV-rank_cross-time.png`: Plot of the cross-validation of the rank in the RRR model.

**Submodules**:

- `analyses.rrr`: Module containing the RRRR function for calculating the RRR model.
- `utils.data_io`: Module for loading and saving data.
- `utils.plots`: Module for plotting data.
"""

import numpy as np
from matplotlib import pyplot as plt

from analyses.rrr import RRRR, cross_time_rrr_coeffs
from utils.data_io import load_pickle, save_pickle
from utils.plots import cv_rank_time_plot

# Load V1 and V2 activity
X = load_pickle("5_block_VISp-activity", path="data/area-responses") # shape (Neurons, Trials, Time)
Y = load_pickle("5_block_VISl-activity", path="data/area-responses") # shape (Neurons, Trials, Time)

def calculate_something():
    '''
    Calculate the time lag between two time series.
    '''

    # Create a results array
    results = np.zeros((len(cv), len(ranks)))

    # Loop through the cross-validation and rank
    for i, c in enumerate(cv):
        for j, r in enumerate(ranks):
            result = cross_time_rrr_coeffs(X, Y, c, r)
            results[i, j] = result['test_score'].mean()

    # Cut off the negative values
    results[results < -0] = np.nan

    print(results)
    
    return results

import yaml

# Load the parameters
params = yaml.safe_load(open('params.yaml'))['rrr-cv-rank-time']

# Define the cross-validation, rank, and time
cv = params['cv']
ranks = params['ranks']
duration = params['duration']
time_bin = params['time-bin']
time_step = params['time-step']

# Create a figure with ncols=time_step/duration and nrwos=time_bin/duration
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
# Print the time
print(f'Calculating time lag')

# Load the results
# result = load_pickle(f'CV-rank')

# Calculate the results
result = calculate_something()
max = np.nanmax(result)

# Save the results
save_pickle(result, f'CV-rank')

# Print the best result
print('Max:', max)

# Plot the results
cv_rank_time_plot(result, cv, ranks, title=f'CV-rank', ax=ax, max=None)

# Save the figure
fig.savefig(f'figures/CV-rank_cross-time.png')
