# layer-interaction-plot.py

"""
This module contains tools for plotting the results of the layer interaction analysis on neural data.

**Parameters**:

- `load`: Session and stimulus-block from load parameters.
- `preprocess`: Preprocess parameters.
- `crosstime`: Crosstime parameters.

**Output**:

- `layer-interaction_<originArea>-to-<targetArea>`: Layer interaction results for each layer combination between the two areas.
"""

import numpy as np
import yaml

from utils import plots
from utils.data_io import load_pickle
from utils.megaplot import megaplot

session = yaml.safe_load(open("params.yaml"))["load"]['session']
blockNumber = yaml.safe_load(open("params.yaml"))["load"]['stimulus-block']
preprocess = yaml.safe_load(open("params.yaml"))["preprocess"]
crosstime = yaml.safe_load(open("params.yaml"))["crosstime"]
timepoints = np.arange(0, preprocess['stimulus-duration'], crosstime['scaling-factor']/1000)  # in seconds
print(f"timepoints: {timepoints}")

areaName = {
    'V1': 'VISp',
    'LM': 'VISl'
}
blockNames = {
    2: 'gabor',
    5: 'natural'
}

def get_minmax(results):
    vmin = np.inf
    vmax = -np.inf
    for output in results.keys():
        for input in results[output].keys():
            result = results[output][input]
            vmin = min(vmin, np.nanmin(result))
            vmax = max(vmax, np.nanmax(result))
    return vmin, vmax

# for direction in ['bottom-up', 'top-down']:
for originArea, targetArea in zip(['V1', 'LM'], ['LM', 'V1']):
    print(f"Plotting {originArea} to {targetArea}")
    
    # Load layer-assignments
    layer_assignments_originArea = load_pickle(f'layer-assignments-{areaName[originArea]}', path='data/units')
    layer_assignments_targetArea = load_pickle(f'layer-assignments-{areaName[targetArea]}', path='data/units')
    
    # Count the number of units for each layer in the origin and target areas
    n_units_originArea = {layer: len(layer_assignments_originArea[layer_assignments_originArea == layer]) for layer in layer_assignments_originArea.unique()}
    n_units_targetArea = {layer: len(layer_assignments_targetArea[layer_assignments_targetArea == layer]) for layer in layer_assignments_targetArea.unique()}
    
    # Results
    results = load_pickle(f"layer-interaction_{originArea}-to-{targetArea}")
    output_layers = list(results.keys())
    input_layers = list(results[output_layers[0]].keys())
    
    # Sort the layers by their number in ascending order
    output_layers = sorted(output_layers)
    input_layers = sorted(input_layers)
    
    # Get the minimum and maximum values for the colorbar across all plots
    vmin, vmax = get_minmax(results)
    print(f"vmin: {vmin}, vmax: {vmax}")
    
    # Plot
    plot = megaplot(nrows=len(output_layers), ncols=len(input_layers), title=f"{session} {blockNames[blockNumber]} {originArea} to {targetArea}")
    plot.row_names = [f'{originArea} l{output_layer} (n={n_units_originArea[output_layer]})' for output_layer in output_layers]
    plot.col_names = [f'{targetArea} l{input_layer} (n={n_units_targetArea[input_layer]})' for input_layer in input_layers]
    for y, output in enumerate(results.keys()):
        for x, input in enumerate(results[output].keys()):
            # print(f"l{output} -> l{input}: {results[output][input]}")
            ax = plot[y, x]
            result = results[output][input]
            nUnits = result.shape[0]
            tick_frequency = int(0.25 / preprocess['step-size'])  # every 250 ms
            imshow = plots.crosstime_RRR(ax, result, originArea, targetArea, timepoints[timepoints < 0.200], vlim=(vmin, vmax), tick_frequency=tick_frequency)
            ax.grid(True)
    
    # Add colorbar
    plot.add_colorbar(imshow)
    
    # Save
    plot.save(f"layer-interaction_{originArea}-to-{targetArea}", path='figures')
    plot.save(f"layer-interaction_{originArea}-to-{targetArea}_{session}_{blockNames[blockNumber]}", path='cache')
    del plot
