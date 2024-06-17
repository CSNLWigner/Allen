



import numpy as np
import yaml

from utils import plots
from utils.data_io import load_pickle
from utils.megaplot import megaplot

session = yaml.safe_load(open("params.yaml"))["load"]['session']
preprocess = yaml.safe_load(open("params.yaml"))["preprocess"]
timepoints = np.arange(0, preprocess['stimulus-duration'], preprocess['step-size'])  # in seconds

# for direction in ['bottom-up', 'top-down']:
for originArea, targetArea in zip(['V1', 'LM'], ['LM', 'V1']):
    
    # Results
    results = load_pickle(f"layer-interaction_{originArea}-to-{targetArea}")
    output_layers = list(results.keys())
    input_layers = list(results[output_layers[0]].keys())
    
    # Get the minimum and maximum values for the colorbar across all plots
    vmin = np.inf
    vmax = -np.inf
    for y, output in enumerate(results.keys()):
        for x, input in enumerate(results[output].keys()):
            result = results[output][input]
            vmin = min(vmin, result.min())
            vmax = max(vmax, result.max())
    
    # Plot
    plot = megaplot(nrows=len(output_layers), ncols=len(input_layers), title=f"{originArea} to {targetArea}")
    plot.row_names = [f'{originArea} l{output_layers}' for output_layers in output_layers]
    plot.col_names = [f'{targetArea} l{input_layers}' for input_layers in input_layers]
    for y, output in enumerate(results.keys()):
        for x, input in enumerate(results[output].keys()):
            # print(f"l{output} -> l{input}: {results[output][input]}")
            ax = plot[y, x]
            result = results[output][input]
            imshow = plots.crosstime_RRR(ax, result, 'LM', 'V1', timepoints[timepoints < 0.200], vlim=(vmin, vmax))
    
    # Add colorbar
    plot.add_colorbar(imshow)
    
    # Save
    plot.save(f"layer-interaction_{originArea}-to-{targetArea}", path='figures')
    plot.save(f"layer-interaction_{originArea}-to-{targetArea}_{session}", path='cache')
    del plot