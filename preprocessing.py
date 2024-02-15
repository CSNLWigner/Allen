from sklearn.preprocessing import StandardScaler
from analyses.data_preprocessing import calculate_residual_activity, get_area_responses, min_max_normalize, z_score_normalize, stimulus_log
from utils.download_allen import cache_allen
from utils.data_io import save_pickle
import yaml

# Load parameters
params = yaml.safe_load(open('params.yaml'))['preprocess']

# An arbitrary session from the Allen Neuropixel dataset
session_id = 1064644573  # 1052533639
cache = cache_allen()
session = cache.get_ecephys_session(ecephys_session_id=session_id)

for area in params['areas']:
    
    # Get the responses for the area
    full_activity = get_area_responses(session, area, session_block=params['stimulus-block'], log=False)
        
    # Get residual activity
    residual_activity = calculate_residual_activity(full_activity) # Neuron-wise AND time-wise
    
    # Normalize the responses
    normalized_activity = z_score_normalize(residual_activity, dims=(0,1,2)) # TODO: Normalize based on ITI activity?
    
    # Sklearn alternative for normalization
    # scaler = StandardScaler()
    # normalized_activity = scaler.fit_transform(residual_activity.reshape(residual_activity.shape[0], -1).T).T.reshape(residual_activity.shape)
    
    # Save the residual activity
    save_pickle(normalized_activity,
                f'{params["stimulus-block"]}_block_{area}-activity', path='data/area-responses')

