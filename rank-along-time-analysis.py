# rank-along-time-analysis.py

"""
This module performs rank-along-time analysis on the Allen Neuropixel dataset.

**Parameters**:

- `load`: Load parameters.
- `preprocess`: Preprocess parameters.
- `rrr`: RRR parameters.
- `rrr-param-search`: RRR parameter search.

**Input**:

- `data/raw-area-responses/<stimulus-block>_block_<predictor>-activity.pickle`: Predictor activity.
- `data/raw-area-responses/<stimulus-block>_block_<target>-activity.pickle`: Target activity.

**Output**:

- `results/max-ranks.pickle`: Optimal rank along time.

**Submodules**:

- `analyses.data_preprocessing`: Data preprocessing.
- `analyses.rrr`: Reduced Rank Regression.
- `utils.data_io`: Data I/O.
"""

import numpy as np
import yaml

from analyses.data_preprocessing import preprocess_area_responses
from analyses.rrr import RRRR
from utils.data_io import load_pickle, save_pickle

# Load parameters
load = yaml.safe_load(open('params.yaml'))['load']
preproc = yaml.safe_load(open('params.yaml'))['preprocess']
rrr_params = yaml.safe_load(open('params.yaml'))['rrr']
search_params = yaml.safe_load(open('params.yaml'))['rrr-param-search']

# Load the activity
full_activity_predictor = load_pickle(
    f'{load["stimulus-block"]}_block_{rrr_params["predictor"]}-activity', path='data/raw-area-responses')
full_activity_target = load_pickle(
    f'{load["stimulus-block"]}_block_{rrr_params["target"]}-activity', path='data/raw-area-responses')

# Define the parameters
cv = rrr_params['cv']
time_lag = preproc['lag-time']
ranks = search_params['rank']

# Define the timepoints
timepoints = np.arange(0, preproc['stimulus-duration'], preproc['step-size']) # in seconds
timepoint_indices = [int(t / preproc['step-size'] / 1000) for t in timepoints]
time_length = int(preproc['stimulus-duration'] / preproc['step-size'])

# Preprocess the area responses
predictor = preprocess_area_responses(full_activity_predictor)
target = preprocess_area_responses(full_activity_target)

# Initialize the list for the results
max_ranks = []

for t, time in zip(timepoint_indices, timepoints):
    results = []
    for k, r in enumerate(ranks):

        # Reduced Rank Regression
        # print(f'Cross-validation: {c}, Time lag: {lag}')
        # result = RRRR(V1.mean(axis=0), V2.mean(axis=0), params['rank'], cv=c) # cross-time RRRR
        result = RRRR(predictor[:, :, t].T,
                      target[:, :, t].T, rank=r, cv=cv, success_log=False)
        
        results.append(result['test_score'].mean())

    # Get the index of the maximum value
    max_idx = np.nanargmax(np.array(results))

    # Save the result averaged over the folds
    max_ranks.append(ranks[max_idx])

# Save the results
save_pickle(max_ranks, 'max-ranks')