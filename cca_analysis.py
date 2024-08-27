# cca_analysis.py

"""
This module makes a CCA analysis across two brain region (VISp and VISpm) from the allen cache databese using allensdk cache and the data/.vbn_s3_cache

The functions for the analysis are in the cca.py and in the utils folder (e.g. download_allen.py and neuropixel.py)

**Parameters in code**:

- `session_id`: The index of the session to load.

**Input**:

None

**Output**:

- `results/cca.pickle`: Pickle file containing the results of the CCA analysis.

**Submodules**:

- `analyses.cca`: Module for the CCA analysis.
- `utils.data_io`: Module for loading and saving data.
- `utils.download_allen`: Module for downloading data from the Allen Institute API.

"""

# from analyses.cca import compare_VISp_VISpm_with_CCA, cca_plot
from analyses import cca
from utils.data_io import save_dict_items
from utils.download_allen import cacheData

# An arbitrary session from the Allen Neuropixel dataset
session_id = 1064644573  # 1052533639
cache = cacheData()
session = cache.get_ecephys_session(ecephys_session_id=session_id)

# Compare VISp and VISpm areas with CCA
result = cca.compare_two_areas(session, 'VISp', 'VISl')

# Save results in the results folder
save_dict_items(result, "cca", path="results")