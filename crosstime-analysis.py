# Import
import numpy as np
import yaml
from analyses.data_preprocessing import preprocess_area_responses
from analyses.rrr import RRRR
from utils.data_io import load_pickle, save_pickle

"""
cross-time analysis based on timpoints of rrr-param-search lag
"""

# Parameters
load = yaml.safe_load(open("params.yaml"))["load"]
preprocess = yaml.safe_load(open("params.yaml"))["preprocess"]
rrr = {**yaml.safe_load(open("params.yaml"))
       ["rrr"], **yaml.safe_load(open("params.yaml"))["best-rrr-params"]}
search = yaml.safe_load(open("params.yaml"))["rrr-param-search"]

# Load raw data
full_predictor = load_pickle(f"{load['stimulus-block']}_block_{rrr['predictor']}-activity", path="data/raw-area-responses")
full_target    = load_pickle(f"{load['stimulus-block']}_block_{rrr['target']}-activity",    path="data/raw-area-responses")

N, K, T = full_predictor.shape
# time_window = preprocess["stimulus-duration"]
# bin_size = preprocess["bin-size"]
time_bin = int(preprocess["bin-size"] * 1000) # in ms

# Define the parameters
prediction_direction = 'top-down' if rrr['predictor'] == 'VISl' else 'bottom-up'
session = load['session']
cv = rrr[session][prediction_direction]['cv']
rank = rrr[session][prediction_direction]['rank']

# timeseries = np.arange(0, preprocess["stimulus-duration"], search['lag']/1000)
timeseries = np.array(search['lag'])

# Add the first timepoint to each element in timeseries
timeseries = timeseries + search['timepoints'][0]

# Init results
results = np.full((len(timeseries), len(timeseries)), fill_value=np.nan)

for x, t_x in enumerate(timeseries):
    for y, t_y in enumerate(timeseries):
        
        # Preprocess the data (the trial duration is only one bin now).
        predictor = preprocess_area_responses(full_predictor[:, :, t_x : t_x + time_bin], stimulus_duration=preprocess["bin-size"], step_size=0.100).squeeze()
        target    = preprocess_area_responses(full_target   [:, :, t_y : t_y + time_bin], stimulus_duration=preprocess["bin-size"], step_size=0.100).squeeze()
        
        # Calculate the RRRR
        model = RRRR(predictor[:, :].T, target[:, :].T, rank=rank, cv=cv, success_log=False)
        
        # Save results
        results[x, y] = model['test_score'].mean()

# Save the results
save_pickle(results, "cross-time-RRR")
        