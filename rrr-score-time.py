# rrr-score-time.py

"""
This module performs rank-along-time analysis on the Allen Neuropixel dataset.

**Parameters**:

- `preprocess`: Preprocess parameters.
- `rrr`: RRR parameters.

**Input**:

- `data/area-responses/5_block_<predictor>-activity.pickle`: Predictor activity.
- `data/area-responses/5_block_<target>-activity.pickle`: Target activity.

**Output**:

- `results/rrr-score-time.pickle`: RRR score along time.

**Submodules**:

- `analyses.rrr`: Module containing the RRRR function for calculating the RRR model.
- `utils.data_io`: Module for loading and saving data.

"""

import yaml
from scipy.stats import sem as standard_error

from analyses.rrr import RRRR
from utils.data_io import load_pickle, save_pickle

# Load the params
preprocess = yaml.safe_load(open('params.yaml'))['preprocess']
rrr = yaml.safe_load(open('params.yaml'))['rrr']

# Load the data
predictor = load_pickle(f"5_block_{rrr['predictor']}-activity", path="data/area-responses") # shape (Neurons, Trials, Time)
target = load_pickle(f"5_block_{rrr['target']}-activity", path="data/area-responses") # shape (Neurons, Trials, Time)

# Get the shape of the data
N_predictor, K, T = predictor.shape

# Initialize the result
rrr_result = []
mean = []
sem  = []

# Loop through the time steps
for time in range(T):
    
    # Get the time step
    predictor_t = predictor[:, :, time].T
    target_t = target[:, :, time].T
    
    # Perform RRR
    result = RRRR(predictor_t, target_t, rank=rrr['rank'], cv=rrr['cv'])['test_score']
    
    # Save the mean and standard error
    mean.append(result.mean())
    sem.append(standard_error(result))

# Save the result
save_pickle({'rrr-result': rrr_result,
             'mean': mean,
             'sem': sem},
             "rrr-score-time")
