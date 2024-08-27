# pca-analysis.py

"""
This module performs PCA analysis on the Allen Neuropixel dataset.

**Parameters**:

- `preprocess`: Preprocess parameters.

**Input**:

- `data/area-responses/<stimulus-block>_block_VISp-activity.pickle`: V1 activity.
- `data/area-responses/<stimulus-block>_block_VISl-activity.pickle`: LM activity.

**Output**:

- `results/pca-analysis.pickle`: PCA analysis results.

**Submodules**:

- `analyses.pca`: Module containing the PCA function for calculating the PCA model.
- `utils.data_io`: Module for loading and saving data.

"""

import yaml
from sklearn.decomposition import PCA

from analyses.pca import pca
from utils.data_io import load_pickle, save_pickle
from utils.utils import iterate_dimension

# Load params
params = yaml.safe_load(open('params.yaml'))['preprocess']

# Define the areas
areas = ['VISp', 'VISl']

# Define the stimuli
# stimuli = zip([2,4,5], ['gabor_filter', 'flashes', 'natural_scenes'])

# Define  the results
results = {}

# Loop over the areas
for area in areas:
    
    # Define the results for the stimuli
    # stim_results = {}
    
    # Loop over the stimuli
    # for stimulus_no, stimulus_name in stimuli:

    # Load data from data/area-responses folder
    neural_activity = load_pickle(f"5_block_{area}-activity", path="data/area-responses")

    # Define the results for the area
    area_results = {}

    # Get the dimensions of the data
    N, K, T = neural_activity.shape
    
    # Loop over the time dimension
    for t, neural_avtivity_slice in iterate_dimension(neural_activity, 2):
        
        # Calculate the time_bin
        time_bin = int(t * params['bin-size'] * 1000)

        # Get the results of PCA
        pca_results = pca(neural_avtivity_slice)
        
        # Save the results
        area_results[time_bin] = pca_results
    
    # Save the results for the area
    results[area] = area_results

# Print the results
# print(results['VISp'][0]['reduced_data'].shape)
# print(results['VISp'][0]['explained_variance_ratio'])
print(results['VISp'][0]['principal_components'].shape)

# Save the results
save_pickle(results, f"pca-analysis", path="results")
