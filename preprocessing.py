import numpy as np
from sklearn.preprocessing import StandardScaler
from analyses.data_preprocessing import calculate_residual_activity, min_max_normalize, preprocess_area_responses, recalculate_neural_activity, z_score_normalize
from utils.download_allen import cache_allen
from utils.data_io import load_pickle, save_pickle
import yaml

# Load parameters
load = yaml.safe_load(open('params.yaml'))['load']
params = yaml.safe_load(open('params.yaml'))['preprocess']

for area in params['areas']:

    # Get the full activity for the area
    full_activity = load_pickle(
        f'{load["stimulus-block"]}_block_{area}-activity', path='data/raw-area-responses')
    
    # Move the activity of V2 back in time by the actual time lag
    if area == params['lag-area']:
        full_activity = np.roll(full_activity, -params['lag-time'], axis=2)
    
    normalized_activity = preprocess_area_responses(full_activity, method='z-score')
    
    # Sklearn alternative for normalization
    # scaler = StandardScaler()
    # normalized_activity = scaler.fit_transform(residual_activity.reshape(residual_activity.shape[0], -1).T).T.reshape(residual_activity.shape)
    
    # Save the residual activity
    save_pickle(normalized_activity,
                f'{load["stimulus-block"]}_block_{area}-activity', path='data/area-responses')

