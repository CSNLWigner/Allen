# Cortical Layer Assignment Analysis

import time

import yaml

from utils.allen_cache import cache_allen
from utils.ccf_volumes import cortical_layer_assignment
from utils.neuropixel import AllenTables

# Load parameters
params = yaml.safe_load(open('params.yaml'))['load']

# An arbitrary session from the Allen Neuropixel dataset
session_id = params['session']  # 1064644573  # 1052533639
cache = cache_allen()

# Create the tables object
tables = AllenTables(cache, session_id)

# Get the units and their layer assignment
units = cortical_layer_assignment(tables.channels, tables.units)
tables.units = units

# Get the units that are assigned to a layer (no nan)
layer_assigned_units = units[units['layer'].notna()]

# Get the length of the layer assigned units and the total number of units
layer_assigned_units_len = len(layer_assigned_units)
total_units_len = len(units)

# Print the results
print('Layer assigned units:', layer_assigned_units_len)
print('Total units:', total_units_len)
print('Percentage:', layer_assigned_units_len / total_units_len * 100)
print()
print(layer_assigned_units)