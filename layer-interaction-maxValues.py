# layer-interaction-maxValues.py

"""
This module acquires the maximum values for each layer interaction from the Allen Neuropixel dataset and appends them to the results/maxValues.csv file.

**Parameters**:

- `load`:
    - `session`: The session to analyze.
- `crosstime`:
    - `scaling-factor`: The scaling factor for the time values.
    
**Input**:

- `results/layer-interaction_<originArea>-to-<targetArea>.pickle`: Pickle file containing the maximum values for each layer interaction from the origin area to the target area.

**Output**:

- `results/maxValues.csv`: CSV file containing the maximum values for each layer interaction.

**Submodules**:

- `analyses.layer_interaction_maxValues`: Module for analyzing the maximum values for each layer interaction.
- `utils.data_io`: Module for loading and saving data.

"""

import warnings

import numpy as np
import pandas as pd
import yaml

from analyses.layer_interaction_maxValues import (find_max_value,
                                                  getGlobalMaxima,
                                                  getUnitsNumber, update_csv)
from utils.data_io import load_pickle

# Ignore FutureWarning
warnings.filterwarnings(action='ignore', category=FutureWarning)

session = yaml.safe_load(open("params.yaml"))["load"]['session']
timeBin = yaml.safe_load(open("params.yaml"))["crosstime"]['scaling-factor']

# for direction in ['bottom-up', 'top-down']:
for originArea, targetArea in zip(['V1', 'LM'], ['LM', 'V1']):
    print(f"Stats from {originArea} to {targetArea}")
    
    # Results
    data = load_pickle(f"layer-interaction_{originArea}-to-{targetArea}") # result has a shape of (timepoints X timepoints) for each layer-pair (output-input)
    
    # Areas Of Interest
    AOI = {
        'first': slice(0, int(100/timeBin)),
        'second':   slice(int(100/timeBin), int(200/timeBin))
    }
    
    nUnits_orig = getUnitsNumber(originArea)
    nUnits_targ = getUnitsNumber(targetArea)

    # Init pandas dataframe
    df = pd.DataFrame(columns=['session', 'direction', 'slice', 'output layer', 'input layer', 'max value', 'x', 'y', 'mean value', 'origin area', 'target area', 'output layer units', 'input layer units'])

    # Calculate the maximum values for each layer-pair
    for slice_name, slice_index in AOI.items():

        # Get the maximum values for each layer-pair
        for output in data.keys():
            for input in data[output].keys():
                dataSlice = data[output][input][slice_index, slice_index]

                # Check if all values in dataSlice are NaN
                if np.all(np.isnan(dataSlice)):
                    maxVal, maxInd = np.nan, (np.nan, np.nan)  # Placeholder for indices, adjust as needed
                    mean_val = np.nan
                else:
                    maxVal, maxInd_raw = find_max_value(dataSlice)
                    maxInd = tuple([(i + slice_index.start) * timeBin for i in maxInd_raw])
                    if slice_name == 'second' and (maxInd[0] < 100 or maxInd[1] < 100):
                        print(f"maxInd: {maxInd} (raw: {maxInd_raw}, slice: {slice_index})")
                    mean_val = np.nanmean(dataSlice)
                    
                # Store the results in a dictionary
                result_dict = {}

                # get the maximum value and index
                result_dict['session'] = session
                result_dict['direction'] = f"{originArea}-to-{targetArea}"
                result_dict['slice'] = slice_name
                result_dict['output layer'] = output
                result_dict['input layer'] = input
                result_dict['max value'] = maxVal
                result_dict['x'] = maxInd[0]
                result_dict['y'] = maxInd[1]
                result_dict['mean value'] = mean_val
                
                # Additional information
                result_dict['origin area'] = originArea
                result_dict['target area'] = targetArea
                result_dict['output layer units'] = nUnits_orig[output]
                result_dict['input layer units'] = nUnits_targ[input]
                
                # Append the results to the dataframe
                df = df.append(result_dict, ignore_index=True)

    # Update the CSV file
    update_csv(df, f"results/maxValues.csv", force=True)
