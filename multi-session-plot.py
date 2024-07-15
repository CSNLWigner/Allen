
# Import the necessary libraries
from utils.data_io import load_pickle, save_pickle
import numpy as np
import yaml
from matplotlib import pyplot as plt

# Load the params
params = yaml.safe_load(open('params.yaml'))['multi-session-thing']

# Define the function
def multi_session_thing(sessions, params):
    """
    Perform a multi-session analysis on the given sessions.

    Args:
        sessions (list): A list of session names or identifiers.
        params (dict): A dictionary containing the parameters for the analysis.

    Returns:
        numpy.ndarray: A 3-dimensional array containing the summed results for each session.
    """
    
    # Determine the area of interest in the data
    area_of_interest = (
        range(params['predictor_A'], params['predictor_B']),
        range(params['target_A'], params['target_B'])
    )
    shape_of_interest = (len(area_of_interest[0]), len(area_of_interest[1]))

    # Initiate the results based on the area of interest
    summed_results = np.zeros(shape_of_interest, len(sessions))

    # Load the data iteratively from the specified path
    for i, session in enumerate(sessions):
        
        # Load the data
        summed_results[:,:,i] = load_pickle(f'cross-time-RRR_{session}', path='fromm3/crosstime')[area_of_interest]
        
    return summed_results




# Load the parameters
times = params['times']

# Define the sessions
sessions = []

# Subplots
fig, axs = plt.subplots(1, len(times))

for time_window in times:
    
    ax = ax[0, times.index(time_window)]
    
    for prediction_direction in ['top-down', 'bottom-up']:
        
        # Load the results
        results = multi_session_thing(sessions, params)
        
        # Save the results
        save_pickle(results, f'{prediction_direction}_cross-time-RRR_{time_window}', path='cache')