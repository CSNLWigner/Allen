import numpy as np
import yaml
from sklearn.preprocessing import StandardScaler

from analyses.data_preprocessing import preprocess_area_responses
from utils.data_io import load_pickle, save_pickle

# Load parameters
load = yaml.safe_load(open('params.yaml'))['load']
params = yaml.safe_load(open('params.yaml'))['preprocess']

# Get the image names
image_names = load_pickle(
    f'{load["stimulus-block"]}_block_image-names', path='data/stimulus-presentations')

for area in params['areas']:

    # Get the full activity for the area
    full_activity = load_pickle(
        f'{load["stimulus-block"]}_block_{area}-activity', path='data/raw-area-responses')
    
    # Move the activity of V2 back in time by the actual time lag
    if area == params['lag-area']:
        full_activity = np.roll(full_activity, -params['lag-time'], axis=2)
    
    normalized_activity = preprocess_area_responses(full_activity, image_names, method='z-score')
    
    # Save the residual activity
    save_pickle(normalized_activity,
                f'{load["stimulus-block"]}_block_{area}-activity', path='data/area-responses')

