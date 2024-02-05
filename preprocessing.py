
import numpy as np
from analyses.data_preprocessing import calculate_residual_activity, get_area_responses, stimulus_log
from utils.download_allen import cache_allen
from utils.data_io import save_pickle
import yaml

from utils.neuropixel import get_area_units, get_unit_responses, makePSTH

params = yaml.safe_load(open('params.yaml'))['preprocess']

# An arbitrary session from the Allen Neuropixel dataset
session_id = 1064644573  # 1052533639
cache = cache_allen()
session = cache.get_ecephys_session(ecephys_session_id=session_id)

for area in params['areas']:
    
    # Get the responses for the area
    full_activity = get_area_responses(session, area, session_block=params['stimulus-block'], log=False)
    
    # Get residual activity
    residual_activity = calculate_residual_activity(full_activity)
    
    # Save the residual activity
    save_pickle(residual_activity, f'{params["stimulus-block"]}_block_{area}-residual-activity', path='data/area-responses')

