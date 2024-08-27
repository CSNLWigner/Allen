# rrr-rank-analysis.py

"""
This module performs rank-along-time analysis on the Allen Neuropixel dataset.

**Parameters**:

- `load`: Load parameters.
- `preprocess`: Preprocess parameters.
- `rrr`: RRR parameters.

**Input**:

- `data/area-responses/<stimulus-block>_block_VISp-activity.pickle`: V1 activity.
- `data/area-responses/<stimulus-block>_block_VISl-activity.pickle`: LM activity.

**Output**:

- `results/VISp_VISl_cross-time-test-scores.pickle`: RRR score along time.

**Submodules**:

- `analyses.rrr`: Module containing the RRRR function for calculating the RRR model.
- `utils.data_io`: Module for loading and saving data.

"""
import numpy as np
from analyses.rrr import rrr_rank_analysis
from utils.data_io import load_pickle, save_pickle
import yaml

# Load parameters
load = yaml.safe_load(open('params.yaml'))['load']
preprocess = yaml.safe_load(open('params.yaml'))['preprocess']
params = yaml.safe_load(open('params.yaml'))['rrr']

# Load V1 and V2 activity
V1_activity = load_pickle(f'{load["stimulus-block"]}_block_VISp-activity', path='data/area-responses')
V2_activity = load_pickle(f'{load["stimulus-block"]}_block_VISl-activity', path='data/area-responses')

# Set the maximum rank to iterate over
max_rank = 15

cvs = [2, 3, 4]

# Get the number of neurons, trials, and time points
N, K_V1, T = V1_activity.shape

# Init the results array
test_scores = np.full((max_rank, T, cvs[-1]+1), np.nan)

for cv in cvs:
    
    test_scores[:,:,cv] = rrr_rank_analysis(V1_activity, V2_activity, cv=cv, log=False)


# Save the errors
save_pickle(test_scores, f'VISp_VISl_cross-time-test-scores', path='results')