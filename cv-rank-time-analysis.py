# cv-ranking-time-analysis.py

"""
This module performs cross-validation, rank, and time analysis on the Allen Neuropixel dataset.

**Parameters**:

- `load`:
    - `session`: The session to analyze.
- `preprocess`: Preprocess parameters.
- `rrr-cv-rank-time`: RRR cross-validation, rank, and time parameters.

**Input**:

- `data/raw-area-responses/<stimulus-block>_block_VISp-activity.pickle`: Pickle file containing the raw activity for the VISp area.
- `data/raw-area-responses/<stimulus-block>_block_VISl-activity.pickle`: Pickle file containing the raw activity for the VISl area.

**Output**:

- `figures/CV-rank-time_<time_bin>-bin_<time_step>-step.png`: The cross-validation, rank, and time plot.

**Submodules**:

- `analyses.rrr`: Module containing the RRRR function for calculating the RRR model.
- `utils.data_io`: Module for loading and saving data.
- `utils.plots`: Module for plotting data.
- `utils.utils`: Module for utility functions.

"""

import numpy as np
import yaml
from matplotlib import pyplot as plt

from analyses.data_preprocessing import (calculate_residual_activity,
                                         z_score_normalize)
from analyses.rrr import RRRR
from utils.data_io import load_pickle
from utils.plots import cv_rank_time_plot

# Load the parameters
load = yaml.safe_load(open('params.yaml'))['load']
params = yaml.safe_load(open('params.yaml'))['preprocess']

# Load V1 and V2 activity
raw_V1 = load_pickle("5_block_VISp-activity", path="data/raw-area-responses") # shape (Neurons, Trials, Time)
raw_V2 = load_pickle("5_block_VISl-activity", path="data/raw-area-responses") # shape (Neurons, Trials, Time)

def calculate_something(from_time=0.200, to_time=0.250):
    '''
    Calculate the time lag between two time series.
    '''

    # Recalculate the neural activity
    # V1 = recalculate_neural_activity(V1, params['stimulus-duration'], params['bin-size'], params['time-step'], orig_time_step=load['step-size'])
    V1 = raw_V1[:, :, int(from_time/load['step-size']):int(to_time/load['step-size'])].sum(axis=2)
    X = z_score_normalize(calculate_residual_activity(V1[:,:,np.newaxis]), dims=(0, 1)).squeeze().T
    V2 = raw_V2[:, :, int(from_time/load['step-size']):int(to_time/load['step-size'])].sum(axis=2)
    Y = z_score_normalize(calculate_residual_activity(V2[:,:,np.newaxis]), dims=(0, 1)).squeeze().T

    # Create a results array
    results = np.zeros((len(cv), len(ranks)))

    # Loop through the cross-validation and rank
    for i, c in enumerate(cv):
        for j, r in enumerate(ranks):
            result = RRRR(X, Y, r, c)
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
fig, axs = plt.subplots(int(duration/time_step), 1, figsize=(10, 10))

results = []
maxes = []
images = []
for i, (from_time, to_time) in enumerate([(time, time+time_bin) for time in np.arange(0, 0.250, time_step)]):
    
    # Print the time
    print(f'Calculating time lag between {from_time} and {to_time} seconds.')

    # Save the results
    result = load_pickle(f'CV-rank_{int(from_time*1000)}-{int(to_time*1000)}ms')
    
    # Calculate the results
    # result = calculate_something(from_time, to_time)
    maxes.append(np.nanmax(result))
    
    # Save the results
    # save_pickle(result, f'CV-rank_{int(from_time*1000)}-{int(to_time*1000)}ms')
    
    # Append the results
    results.append(result)

max = np.nanmax(maxes)
print('Max:', max)

for i, (from_time, to_time) in enumerate([(time, time+time_bin) for time in np.arange(0, 0.250, time_step)]):
    
    # Get the results
    result = results[i]
    
    # Plot the results
    images.append(cv_rank_time_plot(result, cv, ranks, title=f'CV-rank_{int(from_time*1000)}-{int(to_time*1000)} ms', ax=axs[i], max=None))

# Make acommon colorbar for all the subplots
# cbar = fig.colorbar(images[0], ax=axs, orientation='horizontal', fraction=0.05, pad=0.05)

# Save the figure
fig.savefig(f'figures/CV-rank-time_{time_bin}-bin_{time_step}-step.png')
