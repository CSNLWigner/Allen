# This code will make a CCA analysis across two brain region (VISp and VISpm) from the allen cache databese using allensdk cache and the data/.vbn_s3_cache
# The functions for the analysis are in the cca.py and in the utils folder (e.g. download_allen.py and neuropixel.py) 

# Utile functions:
# from allensdk.brain_observatory.ecephys.visualization import plot_mean_waveforms, plot_spike_counts, raster_plot

from analyses.rrr import compare_two_areas
from utils.download_allen import cache_allen
from utils.data_io import save_dict_items

# An arbitrary session from the Allen Neuropixel dataset
session_id = 1064644573  # 1052533639
cache = cache_allen()
session = cache.get_ecephys_session(ecephys_session_id=session_id)

# Compare VISp and VISpm areas with CCA
result = compare_two_areas(session, 'VISp', 'VISl')

# Save results in the results folder
save_dict_items(result, "rrr")