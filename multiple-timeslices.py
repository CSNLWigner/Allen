
from matplotlib import pyplot as plt
import numpy as np
import yaml
from analyses.rrr_time_slice import bidirectional_time_slice
from utils.data_io import load_pickle, save_fig, save_pickle
from utils.megaplot import megaplot
from utils import plots
import sys

from utils.utils import printProgressBar

# Get the parameters from the command line
crosstime_path = sys.argv[1]

# Import params
load = yaml.safe_load(open("params.yaml"))["load"]
preprocess = yaml.safe_load(open("params.yaml"))["preprocess"]
best_params = yaml.safe_load(open("params.yaml"))["best-rrr-params"]
params = yaml.safe_load(open("params.yaml"))["rrr-time-slice"]

# Load raw data
V1 = load_pickle(
    f"{load['stimulus-block']}_block_VISp-activity", path="data/area-responses")
LM = load_pickle(
    f"{load['stimulus-block']}_block_VISl-activity", path="data/area-responses")

# Define the parameters
session = load['session']
predictor_times = params['predictor-time']
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

# The first two columns are the predictor areas
# The two rows of the first to columns are the target areas
for i, predictor in zip([0, 1], ['V1', 'LM']):
    for j, target in zip([0, 1], ['V1', 'LM']):
        
        # Load the crosstime results from cache/*.pickle
        results = load_pickle(f"{names[predictor][target]}_cross-time-RRR_{session}", path=crosstime_path)
        # results = load_pickle("cross-time-RRR", path="results")
        
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
                
printProgressBar(0, len(predictor_times), prefix = 'RRR analysis:', length = 30)
                
for i, predictor_time in enumerate(predictor_times):
    
    # Get the results from the time-slice analysis
    # results = load_pickle(f"{session}_time-slice_{predictor_time}", path="data/time-slices")
    # results = load_pickle("rrr-time-slice", path="results")
    results = bidirectional_time_slice(load['session'], V1, LM, best_params, predictor_time)
    
    # Plot the results
    ax = plot[:, 2+i]
    plots.rrr_time_slice(ax, results, predictor_time, timepoints,
                         (top_down_color, bottom_up_color),
                         ylim=(.03, .30))  # ylim=(.025, .250)
    
    printProgressBar(i + 1, len(predictor_times), prefix = 'RRR analysis:', length = 30)

# Save the figure
plot.savefig('rrr-cross-time-slice-mega-plot', path='figures')