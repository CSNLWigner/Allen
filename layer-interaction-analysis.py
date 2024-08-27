# layer-interaction-analysis.py

"""
This module contains tools for performing layer interaction analysis on neural data.

**Parameters**:

- `load`: Load parameters.
- `crosstime`: Crosstime parameters.
- `rrr`: RRR parameters.

**Output**:

- `layer-interaction_<originArea>-to-<targetArea>`: Layer interaction results for each layer combination between the two areas.
"""

# Import params
import yaml

from analyses.rrr import crosstime_analysis
from utils.data_io import load_csv, load_pickle, save_pickle
from utils.utils import manager

# Import params
# rrr loaded in analyses.rrr as params (as it is only used there)
load = yaml.safe_load(open("params.yaml"))["load"]
crosstime = yaml.safe_load(open("params.yaml"))["crosstime"]

# Get sessions params
sessions_params = load_csv("session-params-old")
session_params = sessions_params[sessions_params['session'] == load['session']]
print(session_params)

# Load raw data
V1_data = {
    'name': 'V1',
    'activity': load_pickle(f"{load['stimulus-block']}_block_VISp-activity", path="data/raw-area-responses"),
    'layer-assignments': load_pickle('layer-assignments-VISp', path='data/units')
}
LM_data = {
    'name': 'LM',
    'activity': load_pickle(f"{load['stimulus-block']}_block_VISl-activity", path="data/raw-area-responses"),
    'layer-assignments': load_pickle('layer-assignments-VISl', path='data/units')
}

for originArea, targetArea in zip([V1_data, LM_data], [LM_data, V1_data]):
    
    # Init Progress Bar
    iterationProgress = 0
    layer_combinations = len(V1_data['layer-assignments'].unique()) * len(LM_data['layer-assignments'].unique())
    # layer_combinations = len(layerAssignments_V1.unique()) * len(layerAssignments_V2.unique())
    
    # Init results
    results = {}
    
    # Iterate over layer combinations
    for output in originArea['layer-assignments'].unique():
        results[output] = {}
        for input in targetArea['layer-assignments'].unique():
            
            # Print Progress Bar
            # printProgressBar(iterationProgress, layer_combinations, prefix=f'l{output} -> l{input}', length=50)
            
            # Test the undersampling lower boundary of layer 5
            # dataBalancing = 'undersampled' if output == 5 or input == 5 else 'none'
            # print('output:', output, 'input:', input, 'dataBalancing:', dataBalancing)
            
            V1 = originArea['activity'][originArea['layer-assignments'].isin([output]), :, :]
            LM = targetArea['activity'][targetArea['layer-assignments'].isin([input]), :, :]
            direction_params = session_params[session_params['direction'] == 'bottom-up']
            cv = direction_params['cv'].values[0]
            rank = direction_params['rank'].values[0]
            result = crosstime_analysis(V1, LM, cv, rank, scaling_factor=crosstime['scaling-factor']) # dataBalancing=dataBalancing # in case of layer-specific undersampling
            
            # Save the results
            results[output][input] = result
            
            # Update Progress Bar
            iterationProgress += 1
            manager.progress_bar(f'{originArea["name"]} to {targetArea["name"]}', iterationProgress, layer_combinations)
    
    # Save the results
    save_pickle(results, f"layer-interaction_{originArea['name']}-to-{targetArea['name']}")
