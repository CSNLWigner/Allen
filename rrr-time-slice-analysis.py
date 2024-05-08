
# Import the necessary libraries
import yaml
from analyses.rrr_time_slice import bidirectional_time_slice
from utils.data_io import load_pickle, save_pickle
from scipy.stats import sem as SEM

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
