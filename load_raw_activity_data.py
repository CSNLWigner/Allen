from sklearn.preprocessing import StandardScaler
from analyses.data_preprocessing import get_area_responses
from utils.download_allen import cache_allen
from utils.data_io import save_pickle
import yaml

# Load parameters
params = yaml.safe_load(open('params.yaml'))['load']

# An arbitrary session from the Allen Neuropixel dataset
session_id = 1064644573  # 1052533639
cache = cache_allen()
session = cache.get_ecephys_session(ecephys_session_id=session_id)

for area in params['areas']:

    # Get the responses for the area
    full_activity = get_area_responses(
        session, area, session_block=params['stimulus-block'], log=False)

    print(full_activity.shape)

    # Save the residual activity
    save_pickle(full_activity,
                f'{params["stimulus-block"]}_block_{area}-activity', path='data/raw-area-responses')
