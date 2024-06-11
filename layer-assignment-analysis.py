# Cortical Layer Assignment Analysis

import time

import pandas as pd
import yaml

from utils.allen_cache import cache_allen
from utils.ccf_volumes import cortical_layer_assignment
from utils.data_io import save_pickle
from utils.debug import ic
from utils.neuropixel import AllenTables

# Load parameters
params = yaml.safe_load(open('params.yaml'))['load']

# An arbitrary session from the Allen Neuropixel dataset
session_id = params['session']  # 1064644573  # 1052533639
cache = cache_allen()

# Create the tables object
tables = AllenTables(cache, session_id)

# Get the units and their layer assignment DataFrame
layerAssignments = cortical_layer_assignment(tables.channels, tables.units)

# Save the units
for area in params['areas']:
    output = layerAssignments[layerAssignments['structure_acronym'] == area]
    output = output['layer']
    
    save_pickle(output, f'layer-assignments-{area}', path='data/units')

    # Print statistics (unique values and ratio of value 0 to all values)
    unique = output.unique()
    print(f"Unique values in {area}:", unique)
    # print("Ratio of value 0:", output.value_counts()[0] / len(output))
    for i in unique:
        ratio = output.value_counts()[i] / len(output)
        percent = "{:.2%}".format(ratio)
        print(f"Ratio of value {i}:", percent)