import yaml

from analyses.data_preprocessing import transform_behav_data
from utils.data_io import save_pickle
from utils.download_allen import cacheData

# Load parameters
preprocess = yaml.safe_load(open('params.yaml'))['preprocess']

# An arbitrary session from the Allen Neuropixel dataset
session_id = 1064644573  # 1052533639
cache = cacheData()
session = cache.get_ecephys_session(ecephys_session_id=session_id)

params = {
    'running-speed': 'speed',
    'pupil-area': 'pupil_area',
    'lick-times': 'timestamps'
}

sessions = {
    'running-speed': session.running_speed,
    'pupil-area': session.eye_tracking[session.eye_tracking['likely_blink']],
    'lick-times': session.licks
}

behavior_names = ['running-speed', 'pupil-area', 'lick-times']

for name in behavior_names:
    
    # Print the behavior name
    print(f'{name.capitalize()}')
    
    # Get the behavior data and parameter
    data, param = sessions[name], params[name]
    
    # Transform the behavior data
    transformed_data = transform_behav_data(data, param,
                                            session.stimulus_presentations,
                                            preprocess['stimulus-block'],
                                            log=True)
    
    # z-score the data
    transformed_data = (transformed_data - transformed_data.mean()) / transformed_data.std()
    
    # Save the transformed data
    save_pickle(transformed_data, f'{preprocess["stimulus-block"]}_block_{name}', path='data/behav-responses')