# multiple-timeslices.py

"""
This module analyzes the interaction between layers in the Allen Neuropixel dataset.

**Parameters**:

- `load`:
    - `session`: The session to analyze.
- `preprocess`: The preprocess parameters.
- `timeslice`: The time-slice parameters.
- `crosstime`: The cross-time parameters.

**Input**:

- `data/raw-area-responses/<stimulus-block>_block_VISp-activity.pickle`: Pickle file containing the raw activity for the VISp area.
- `data/raw-area-responses/<stimulus-block>_block_VISl-activity.pickle`: Pickle file containing the raw activity for the VISl area.
- `data/area-responses/<stimulus-block>_block_VISp-activity.pickle`: Pickle file containing the preprocessed activity for the VISp area.
- `data/area-responses/<stimulus-block>_block_VISl-activity.pickle`: Pickle file containing the preprocessed activity for the VISl area.
- `data/stimulus-presentations/<stimulus-block>_block_image-names.pickle`: Pickle file containing the stimulus names.

**Output**:

- `figures/rrr-cross-time-slice-mega-plot.png`: The mega plot of the RRR analysis.

**Submodules**:

- `analyses.rrr`: Module containing the RRRR function for calculating the RRR model.
- `analyses.rrr_time_slice`: Module containing the time-slice analysis function.
- `utils.data_io`: Module for loading and saving data.
- `utils.megaplot`: Module for creating mega plots.
- `utils.utils`: Module for utility functions.
- `utils.plots`: Module for plotting functions.

"""

import numpy as np
import yaml
from matplotlib import pyplot as plt

from analyses.rrr import crosstime_analysis
from analyses.rrr_time_slice import bidirectional_time_slice
from utils import plots
from utils.data_io import load_csv, load_pickle, save_fig, save_pickle
from utils.megaplot import megaplot
from utils.utils import printProgressBar

# Get the parameters from the command line
# crosstime_path = sys.argv[1]

# Import params
load = yaml.safe_load(open("params.yaml"))["load"]
preprocess = yaml.safe_load(open("params.yaml"))["preprocess"]
timeslice = yaml.safe_load(open("params.yaml"))["rrr-time-slice"]
crosstime = yaml.safe_load(open("params.yaml"))["crosstime"]

# Get sessions-params.csv
sessions_params = load_csv("sessions-params") # columns: session, predictor, target, direction, r2, time, cv, lag, rank

# Get the rows for the current session
session_params = sessions_params[sessions_params['session'] == load['session']]
print(session_params)

# MARK: - Load data

# Load raw data
raw_V1 = load_pickle(
    f"{load['stimulus-block']}_block_VISp-activity", path="data/raw-area-responses")
raw_LM = load_pickle(
    f"{load['stimulus-block']}_block_VISl-activity", path="data/raw-area-responses")
V1 = load_pickle(
    f"{load['stimulus-block']}_block_VISp-activity", path="data/area-responses")
LM = load_pickle(
    f"{load['stimulus-block']}_block_VISl-activity", path="data/area-responses")
stimuli = load_pickle(
    f"{load['stimulus-block']}_block_image-names", path="data/stimulus-presentations")

# Define the parameters
session = load['session']
predictor_times = timeslice['predictor-time']
timepoints = np.arange(0, preprocess['stimulus-duration'], preprocess['step-size'])  # in seconds
names = {
    'V1': {
        'V1': 'V1',
        'LM': 'bottom-up'
    },
    'LM': {
        'V1': 'top-down',
        'LM': 'LM'
    }
}

# Define the colors
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
top_down_color = colors[0]
bottom_up_color = colors[1]

# Init subplots
plot = megaplot(nrows=2, ncols=2+len(predictor_times), title=session)
plot.rownames(range(2), ['predictor V1', 'predictor LM'])
plot.colnames(range(2+len(predictor_times)),
              ['target V1', 'target LM'] +
              [f"{i} s" for i in predictor_times])

'''
          | predictor:V1 | predictor:LM | time:0 | time:1 | time:2 | time:3 | time:4 | time:5 |
target:V1 | V1 -> V1     | LM -> V1     | slice  | slice  | slice  | slice  | slice  | slice  |
target:LM | V1 -> LM     | LM -> LM     | slice  | slice  | slice  | slice  | slice  | slice  |
'''

# MARK: - Crosstime analysis

# The first two columns are the predictor areas
# The two rows of the first to columns are the target areas
for i, predictor, predictor_activity in zip([0, 1], ['V1', 'LM'], [raw_V1, raw_LM]): # ROWS
    for j, target, target_activity in zip([0, 1], ['V1', 'LM'], [raw_V1, raw_LM]): # COLUMNS
        
        # Extract the session parameters
        if predictor == 'V1':
            session_key = 'bottom-up'
        elif predictor == 'LM':
            session_key = 'top-down'
        direction_params = session_params[session_params['direction'] == session_key]
        cv = direction_params['cv'].values[0]
        rank = direction_params['rank'].values[0]
        
        # Load the crosstime results from cache/*.pickle
        # results = load_pickle(f"{names[predictor][target]}_cross-time-RRR_{session}", path=crosstime_path)
        
        # Calculate crosstime analysis
        results = crosstime_analysis(predictor_activity, target_activity, stimuli, cv, rank, scaling_factor=crosstime['scaling-factor'])
        
        # Plot the results
        ax = plot[i, j]
        plots.crosstime_RRR(ax, results, predictor, target, timepoints[timepoints < 0.200])
        
        # Blue and orange lines on the cross-time plots
        if predictor == 'V1' and target == 'LM':
            for predictor_time in predictor_times:
                ax.axhline(y=predictor_time, color=bottom_up_color, linestyle='-')
        if predictor == 'LM' and target == 'V1':
            for predictor_time in predictor_times:
                ax.axhline(y=predictor_time, color=top_down_color, linestyle='-')

# MARK: - Time-slice analysis
                
printProgressBar(0, len(predictor_times), prefix = 'RRR analysis:', length = 30)
                
for i, predictor_time in enumerate(predictor_times):
    
    # Get the results from the time-slice analysis
    # results = load_pickle(f"{session}_time-slice_{predictor_time}", path="data/time-slices")
    # results = load_pickle("rrr-time-slice", path="results")
    results = bidirectional_time_slice(V1, LM, session_params, predictor_time)
    
    # Plot the results
    ax = plot[:, 2+i]
    plots.rrr_time_slice(ax, results, predictor_time, timepoints,
                         (top_down_color, bottom_up_color),
                         ylim=(.03, .30))  # ylim=(.025, .250)
    
    printProgressBar(i + 1, len(predictor_times), prefix = 'RRR analysis:', length = 30)

# Save the figure
plot.savefig('rrr-cross-time-slice-mega-plot', path='figures')