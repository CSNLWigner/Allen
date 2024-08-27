# rrr-time-slice-analysis.py

"""
This module performs bidirectional time-slice analysis on the Allen Neuropixel dataset.

The script loads the preprocessed activity for the VISp and VISl areas from the `data/area-responses` folder. It then performs the bidirectional time-slice analysis using the `bidirectional_time_slice` function from the `analyses.rrr_time_slice` module. The best parameters for the RRR model are loaded from the `params.yaml` file.

**Parameters**:

- `load`: Load parameters.
- `best-rrr-params`: Best RRR parameters.
- `rrr-time-slice`: Time-slice parameters.

**Input**:

- `data/area-responses/<stimulus-block>_block_VISp-activity.pickle`: Pickle file containing the preprocessed activity for the VISp area.
- `data/area-responses/<stimulus-block>_block_VISl-activity.pickle`: Pickle file containing the preprocessed activity for the VISl area.

**Output**:

- `results/rrr-time-slice.pickle`: Pickle file containing the results of the time-slice analysis.

**Submodules**:

- `analyses.rrr_time_slice`: Module containing the time-slice analysis function.
- `utils.data_io`: Module for loading and saving data.

"""
# Import the necessary libraries
import yaml
from analyses.rrr_time_slice import bidirectional_time_slice
from utils.data_io import load_pickle, save_pickle

# Import params
load = yaml.safe_load(open("params.yaml"))["load"]
best_params = yaml.safe_load(open("params.yaml"))["best-rrr-params"]
params = yaml.safe_load(open("params.yaml"))["rrr-time-slice"]

# Load raw data
V1_activity = load_pickle(
    f"{load['stimulus-block']}_block_VISp-activity", path="data/area-responses")
LM_activity = load_pickle(
    f"{load['stimulus-block']}_block_VISl-activity", path="data/area-responses")

# Raise error if the predictor-time is not a float
if not isinstance(params['predictor-time'], float):
    raise ValueError("The predictor-time must be a float.")

# Perform the analysis
results = bidirectional_time_slice(load['session'],
    V1_activity, LM_activity,
    best_params, params['predictor-time'])

# Save the results
save_pickle(results, f"rrr-time-slice")
