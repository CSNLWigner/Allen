import yaml

from analyses.data_preprocessing import get_area_responses
from utils.data_io import save_pickle
from utils.download_allen import cacheData

# Load parameters
params = yaml.safe_load(open('params.yaml'))['load']

# An arbitrary session from the Allen Neuropixel dataset
session_id = params['session'] # 1064644573  # 1052533639
cache = cacheData()
try:
    # session = cache.get_ecephys_session(ecephys_session_id=session_id)
    session = cache.get_session_data(
              cache.get_session_table().index.values[session_id])
except Exception as e:
    # print("Session not found. Check the session ID. Here are the available sessions:")
    # sessions = cache.get_session_table()
    # print(sessions.index.values)
    n_sessions = len(cache.get_session_table())
    print(f"Session not found. Check the session ID. There are {n_sessions} available sessions.")
    raise e

# Get the units
units = session.units

# print('\n'.join([attr_or_method for attr_or_method in dir(
#     units) if attr_or_method[0] != '_']))

# print(session.stimulus_presentations.stimulus_name.unique())

stim_table = session.stimulus_presentations[session.stimulus_presentations.stimulus_name == params['stimulus-block']]

for area in params['areas']:
    
    # Get the responses for the area
    full_activity = get_area_responses(session, area, stim_table, units, log=False)

    print(full_activity.shape)

    # Save the residual activity
    save_pickle(full_activity,
                f'{params["stimulus-block"]}_block_{area}-activity',
                path='data/raw-area-responses')
