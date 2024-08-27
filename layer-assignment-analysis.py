# layer-assignment-analysis.py

"""
This module assigns the cortical layers to the units in the Allen Neuropixel dataset.

**Parameters**:

- `load`:
    - `areas`: A list of brain areas to analyze.

**Input**:

None

**Output**:

- `data/units/layer-assignments-<area>.pickle`: Pickle files containing the layer assignments for each brain area.

**Submodules**:

- `utils.download_allen`: Module for downloading data from the Allen Institute API.
- `utils.ccf_volumes`: Module for extracting the cortical layer assignment.
- `utils.data_io`: Module for loading and saving data.
- `utils.neuropixel`: Module for extracting the units from the Allen Neuropixel dataset.

"""


import yaml

from utils.download_allen import cacheData
from utils.ccf_volumes import cortical_layer_assignment
from utils.data_io import save_pickle
from utils.neuropixel import AllenTables, get_area_units

# Load parameters
params = yaml.safe_load(open('params.yaml'))['load']

# An arbitrary session from the Allen Neuropixel dataset
session_id = params['session']  # 1064644573  # 1052533639
cache = cacheData()

# Create the tables object
tables = AllenTables(cache, session_id)

# Get the filtered units for the areas
filteredUnits = get_area_units(tables.units, ['VISp', 'VISl'])

# Get the units and their layer assignment DataFrame
layerAssignments = cortical_layer_assignment(tables.channels, filteredUnits)

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