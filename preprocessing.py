# preprocessing.py

"""
This module preprocesses the area responses for the RRR model.

The script loads the area responses for a given stimulus block. It then z-scores the area responses and saves the normalized activity as a pickle file in the `data/area-responses` directory.

**Parameters**:

- `load`:
    - `stimulus-block`: The name of the stimulus block to analyze.
- `preprocess`:
    - `areas`: A list of brain areas to preprocess.
    - `lag-area`: The brain area to move back in time.
    - `lag-time`: The time lag to apply to the area responses.

**Input**:

- `data/raw-area-responses/<stimulus-block>_block_<area>-activity.pickle`: Pickle files containing the raw activity data for each brain area.

**Output**:

- `data/area-responses/<stimulus-block>_block_<area>-activity.pickle`: Pickle files containing the preprocessed activity data for each brain area.

**Submodules**:

- `analyses.data_preprocessing`: Module for data preprocessing.
- `utils.data_io`: Module for loading and saving data.

"""

import numpy as np
import yaml
from sklearn.preprocessing import StandardScaler

from analyses.data_preprocessing import preprocess_area_responses
from utils.data_io import load_pickle, save_pickle

# Load parameters
load = yaml.safe_load(open('params.yaml'))['load']
params = yaml.safe_load(open('params.yaml'))['preprocess']

for area in params['areas']:

    # Get the full activity for the area
    full_activity = load_pickle(
        f'{load["stimulus-block"]}_block_{area}-activity', path='data/raw-area-responses')
    
    # Move the activity of V2 back in time by the actual time lag
    if area == params['lag-area']:
        full_activity = np.roll(full_activity, -params['lag-time'], axis=2)
    
    normalized_activity = preprocess_area_responses(full_activity, method='z-score')
    
    # Save the residual activity
    save_pickle(normalized_activity,
                f'{load["stimulus-block"]}_block_{area}-activity', path='data/area-responses')

