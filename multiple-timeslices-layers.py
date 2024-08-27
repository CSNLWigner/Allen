# multiple-timeslices-layers.py

"""
This module analyzes the interaction between layers in the Allen Neuropixel dataset.

**Parameters**:

- `load`:
    - `session`: The session to analyze.
- `preprocess`: The preprocess parameters.
- `timeslice`: The time-slice parameters.
- `crosstime`: The cross-time parameters.
- `interaction-layers`: The layers to analyze.

**Input**:

- `data/units/layer-assignments-VISp.pickle`: Pickle file containing the layer assignments for the VISp area.
- `data/units/layer-assignments-VISl.pickle`: Pickle file containing the layer assignments for the VISl area.
- `data/raw-area-responses/<stimulus-block>_block_VISp-activity.pickle`: Pickle file containing the raw activity for the VISp area.
- `data/raw-area-responses/<stimulus-block>_block_VISl-activity.pickle`: Pickle file containing the raw activity for the VISl area.
- `data/area-responses/<stimulus-block>_block_VISp-activity.pickle`: Pickle file containing the preprocessed activity for the VISp area.
- `data/area-responses/<stimulus-block>_block_VISl-activity.pickle`: Pickle file containing the preprocessed activity for the VISl area.
- `data/stimulus-presentations/<stimulus-block>_block_image-names.pickle`: Pickle file containing the stimulus names.

**Output**:

- `figures/rrr-cross-time-slice-mega-plot.png`: The mega plot of the RRR analysis.

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
neurmask = yaml.safe_load(open("params.yaml"))["interaction-layers"]

# Get sessions-params.csv
sessions_params = load_csv("session-params-old") # columns: session, predictor, target, direction, r2, time, cv, lag, rank

# Get the rows for the current session
session_params = sessions_params[sessions_params['session'] == load['session']]
print(session_params)

# MARK: - Load data

# Load the layer assignments
layerAssignments_V1 = load_pickle('layer-assignments-VISp', path='data/units')
layerAssignments_V2 = load_pickle('layer-assignments-VISl', path='data/units')

# Load raw data
raw_V1 = load_pickle(f"{load['stimulus-block']}_block_VISp-activity", path="data/raw-area-responses")
raw_LM = load_pickle(f"{load['stimulus-block']}_block_VISl-activity", path="data/raw-area-responses")
V1 = load_pickle(f"{load['stimulus-block']}_block_VISp-activity", path="data/area-responses")
LM = load_pickle(f"{load['stimulus-block']}_block_VISl-activity", path="data/area-responses")
stimuli = load_pickle(f"{load['stimulus-block']}_block_image-names", path="data/stimulus-presentations")

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

# Bottom-up (V1 -> LM):
ax = plot[0, 1]
V1 = raw_V1[layerAssignments_V1.isin(neurmask['V1']['output']), :, :]
print('V1 output neurons:', V1.shape[0])
LM = raw_LM[layerAssignments_V2.isin(neurmask['LM']['input']), :, :]
print('LM input neurons:', LM.shape[0])
direction_params = session_params[session_params['direction'] == 'bottom-up']
cv = direction_params['cv'].values[0]
rank = direction_params['rank'].values[0]
results = crosstime_analysis(V1, LM, cv, rank, scaling_factor=crosstime['scaling-factor'])
plots.crosstime_RRR(ax, results, 'V1', 'LM', timepoints[timepoints < 0.200])
for predictor_time in predictor_times:
    ax.axhline(y=predictor_time, color=bottom_up_color, linestyle='-')

# Top-down (LM -> V1):
ax = plot[1, 0]
V1 = raw_V1[layerAssignments_V1.isin(neurmask['V1']['input']), :, :]
print('V1 input neurons:', V1.shape[0])
LM = raw_LM[layerAssignments_V2.isin(neurmask['LM']['output']), :, :]
print('LM output neurons:', LM.shape[0])
direction_params = session_params[session_params['direction'] == 'top-down']
cv = direction_params['cv'].values[0]
rank = direction_params['rank'].values[0]
results = crosstime_analysis(LM, V1, cv, rank, scaling_factor=crosstime['scaling-factor'])
plots.crosstime_RRR(ax, results, 'LM', 'V1', timepoints[timepoints < 0.200])
for predictor_time in predictor_times:
    ax.axhline(y=predictor_time, color=top_down_color, linestyle='-')


# MARK: - Time-slice analysis
                
printProgressBar(0, len(predictor_times), prefix = 'RRR analysis:', length = 30)
                
for i, predictor_time in enumerate(predictor_times):
    
    # Get the results from the time-slice analysis
    # results = bidirectional_time_slice(V1, LM, session_params, predictor_time)
    
    # Plot the results
    ax = plot[:, 2+i]
    # plots.rrr_time_slice(ax, results, predictor_time, timepoints,
    #                      (top_down_color, bottom_up_color),
    #                      isWithinSameArea=isWithinSameArea,
    #                      ylim=(.03, .30))  # ylim=(.025, .250)
    
    printProgressBar(i + 1, len(predictor_times), prefix = 'RRR analysis:', length = 30)

# Save the figure
plot.savefig('rrr-cross-time-slice-mega-plot', path='figures')