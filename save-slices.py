
'''
This is instead of plot section of the dvc pipeline
'''

# Import the necessary libraries
from utils.data_io import load_pickle, save_pickle
import yaml
import sys

# Get the parameters from the command line
prediction_direction = sys.argv[1]

# Load the params
load = yaml.safe_load(open('params.yaml'))['load']
session = load['session']
rrr = yaml.safe_load(open('params.yaml'))['rrr']

# Load the results from dir results
results = load_pickle('cross-time-RRR', path='results')

# Save the results in a different loction
save_pickle(results, f'{prediction_direction}_cross-time-RRR_{session}', path='cache')