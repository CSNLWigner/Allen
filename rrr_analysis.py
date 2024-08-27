# rrr_analysis.py

"""
This code will make a CCA analysis across two brain region (VISp and VISpm) from the allen cache databese using allensdk cache and the data/.vbn_s3_cache

The functions for the analysis are in the cca.py and in the utils folder (e.g. download_allen.py and neuropixel.py) 

Utile functions:

- from allensdk.brain_observatory.ecephys.visualization import plot_mean_waveforms, plot_spike_counts, raster_plot

**Parameters**:

- `preprocess`:
    - `stimulus-block`: The name of the stimulus block to analyze.

**Input**:

- `data/area-responses/<stimulus-block>_block_VISp-responses.pickle`: Pickle file containing the responses of the units in the VISp area.
- `data/area-responses/<stimulus-block>_block_VISpm-responses.pickle`: Pickle file containing the responses of the units in the VISpm area.

**Output**:

- `results/rrr.pickle`: Pickle file containing the results of the RRR analysis.

**Submodules**:

- `analyses.rrr`: Module for the RRR analysis.
- `utils.data_io`: Module for loading and saving data.
- `utils.download_allen`: Module for downloading data from the Allen Institute API.

"""

import yaml

from analyses.rrr import compare_two_areas
from utils.data_io import load_pickle, save_dict_items

preprocess = yaml.safe_load(open('params.yaml'))['preprocess']

# Load brain area responses
area_X_responses = load_pickle(f'data/area-responses/{preprocess["stimulus-block"]}_block_VISp-responses')
area_Y_responses = load_pickle(f'data/area-responses/{preprocess["stimulus-block"]}_block_VISl-responses')

# Make RRR analysis
result = compare_two_areas(area_X_responses, area_Y_responses, log=True)

# Save results in the results folder
save_dict_items(result, "rrr", path="results")