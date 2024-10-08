"""
This module calculates the RRR ranks between the layers of V1 and LM.

**Usage**:

```shell
$ python layer-rank-analysis.py [-l]
```

**Arguemnts**:

- `-l`: Log switch.

**Parameters**:

- `load`: Load parameters.
- `preprocess`: Preprocess parameters.
- `layer-rank`: Layer rank parameters.

**Input**:

- `data/area-responses/<stimulus-block>_block_VISp-activity.pickle`: V1 activity.
- `data/area-responses/<stimulus-block>_block_VISl-activity.pickle`: LM activity.
- `data/units/layer-assignments-VISp.pickle`: Layer assignments for V1.
- `data/units/layer-assignments-VISl.pickle`: Layer assignments for LM.

**Output**:

- `results/layer-rank.pickle`: Layer rank results.
- `results/layer-r2.pickle`: Layer R2 results.

**Submodules**:

- `analyses.layer_rank`: Layer rank analysis.
- `utils.data_io`: Data I/O.
- `utils.utils`: Utilities.
"""

import sys

import numpy as np
import yaml

from analyses.layer_rank import calc_ranks
from utils.data_io import load_pickle, save_pickle
from utils.utils import get_args

# Get the arguments
opts, args = get_args(sys.argv)

# Log switch
log = False
if "-l" in opts:
    log = True

# Load parameters
load = yaml.safe_load(open('params.yaml'))['load']
preproc = yaml.safe_load(open('params.yaml'))['preprocess']
# ATTENTION! 'layer-rank' parameters are used in the subfunctions!

# Load the data
V1_data = {
    'name': 'V1',
    'activity': load_pickle(f"{load['stimulus-block']}_block_VISp-activity",
                            path="data/area-responses"),
    'layer-assignments': load_pickle('layer-assignments-VISp',
                                     path='data/units')
}
LM_data = {
    'name': 'LM',
    'activity': load_pickle(f"{load['stimulus-block']}_block_VISl-activity",
                            path="data/area-responses"),
    'layer-assignments': load_pickle('layer-assignments-VISl',
                                     path='data/units')
}

# Print the time
print(f'Calculating ranks for {load["session"]}...')

# Get the timepoints (seconds)
timepoints = np.arange(0, preproc['stimulus-duration'], preproc['step-size'])

# Turn timepoints into indices
time_indeces = (timepoints / preproc['step-size']).astype(int)

# Calculate the ranks
rank_results, r2_results = calc_ranks(V1_data, LM_data, time_indeces, log=log) # Shape: (nAreas(2), nLayers(6+1), nLayers(6+1), nTimepoints)

# Save the results
save_pickle(rank_results, f'layer-rank')
save_pickle(r2_results, f'layer-r2')

