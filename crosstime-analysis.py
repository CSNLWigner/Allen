# Import
import sys

import yaml

from analyses.rrr import RRRR, crosstime_analysis
from utils.data_io import load_pickle, save_pickle

"""
cross-time analysis based on timpoints of rrr-param-search lag
"""

# Get the parameters from the command line
prediction_direction = sys.argv[1]

# Get the predictor and the target from the prediction direction
predictor = 'VISl' if prediction_direction == 'top-down' else 'VISp'
target = 'VISp' if prediction_direction == 'top-down' else 'VISl'

# Parameters
load = yaml.safe_load(open("params.yaml"))["load"]
preprocess = yaml.safe_load(open("params.yaml"))["preprocess"]
rrr = {**yaml.safe_load(open("params.yaml"))
       ["rrr"], **yaml.safe_load(open("params.yaml"))["best-rrr-params"]}
params = yaml.safe_load(open("params.yaml"))["crosstime"]

# Load raw data
full_predictor = load_pickle(f"{load['stimulus-block']}_block_{rrr['predictor']}-activity", path="data/raw-area-responses")
full_target    = load_pickle(f"{load['stimulus-block']}_block_{rrr['target']}-activity",    path="data/raw-area-responses")

# Run crosstime-analysis
results = crosstime_analysis(full_predictor, full_target, rrr, params)

# Save the results
save_pickle(results, f"{prediction_direction}_cross-time-RRR")
        