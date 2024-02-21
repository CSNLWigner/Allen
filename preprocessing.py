from sklearn.preprocessing import StandardScaler
from analyses.data_preprocessing import calculate_residual_activity, min_max_normalize, recalculate_neural_activity, z_score_normalize
from utils.download_allen import cache_allen
from utils.data_io import load_pickle, save_pickle
import yaml

# Load parameters
params = yaml.safe_load(open('params.yaml'))['preprocess']

for area in params['areas']:

    # Get the full activity for the area
    full_activity = load_pickle(
        f'{params["stimulus-block"]}_block_{area}-activity', path='data/raw-area-responses')
    
    # Recalculate time steps and time bins of the full activity
    full_activity = recalculate_neural_activity(full_activity, 
        params['stimulus-duration'], params['step-size'], params['bin-size'],
        orig_time_step=0.005)
    
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

