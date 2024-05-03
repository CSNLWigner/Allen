
import yaml
from analyses.data_preprocessing import preprocess_area_responses
from analyses.rrr_time_slice import RRRR_time_slice
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

N, K, T = V1_activity.shape

# Define the parameters
session = load['session']
abstract_areas = {
    'top-down': {
        'predictor': LM_activity,
        'target': V1_activity
    },
    'bottom-up': {
        'predictor': V1_activity,
        'target': LM_activity
    }
}

# Init results
results = {}

# Iterate through the prediction directions
for prediction_direction in ['top-down','bottom-up']:

    # Extract the data
    predictor_activity = abstract_areas[prediction_direction]['predictor']
    target_activity = abstract_areas[prediction_direction]['target']
    cv = best_params[session][prediction_direction]['cv']
    rank = best_params[session][prediction_direction]['rank']
    
    # Calculate the RRRR
    result = RRRR_time_slice(predictor_activity, target_activity, params['predictor-time'], cv, rank, log=True)
    
    # Save the results
    results[prediction_direction] = result

# Save the results
save_pickle(results, f"rrr-time-slice")
