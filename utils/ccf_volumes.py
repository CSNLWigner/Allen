# utils/ccf_volumes.py

"""
This submodule contains tools for assigning cortical layers to channels and units based on the Allen Brain Atlas Common Coordinate Framework (CCF) volumes.

Created on Tue Apr  7 08:41:07 2020

Author: joshs

References:
- [Cortical Layers using CCFv3 in Neuropixels Data](https://community.brain-map.org/t/cortical-layers-using-ccfv3-in-neuropixels-data/1247/4)
- [CCFv3 Annotation Volume](https://www.dropbox.com/scl/fo/6x7ovegu2jp4jxrhyv0fi/APGHNCbZrJFU6xfccmyu1Vw?dl=0&e=2&rlkey=qqn8efbm4pto0olh0g9o5ctjs)

Functions:
- get_layer_name(acronym) -> int: Get the layer number from the given acronym.
- get_structure_ids(df, annotations) -> np.ndarray: Get the structure IDs for the given DataFrame.
- cortical_depth_calculation(channels) -> pd.DataFrame: Calculate the cortical depth for the given channels.
- layer_assignment_to_channels(channels) -> pd.DataFrame: Assign cortical layers to the given channels.
- cortical_layer_assignment(channels, units) -> pd.DataFrame: Assign cortical layers to the given units.
"""

# %%

import os

import numpy as np
import pandas as pd
from allensdk.brain_observatory.ecephys.behavior_ecephys_session import \
    BehaviorEcephysSession
from allensdk.brain_observatory.ecephys.ecephys_project_cache import \
    EcephysProjectCache

# %%

# cache_dir = 'data/.vbn_s3_cache' 

# manifest_path = os.path.join(cache_dir, "manifest.json")
# cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

# import warnings
# warnings.filterwarnings("ignore")

# sessions = cache.get_session_table()

# %%

# # Print the available data fields
# print(sessions.index.values)

# session = cache.get_session_data(sessions.index.values[0])

# print('Actual session: ', session.ecephys_session_id)

# %%

path = 'data/ccf_volumes/'

import nrrd

streamlines, header = nrrd.read(f'{path}laplacian_10.nrrd')

annotations = np.load(f'{path}annotation_volume_10um_by_index.npy')

structure_tree = pd.read_csv(f'{path}ccf_structure_tree.csv', index_col=0)


# %% Functions

import re


def get_layer_name(acronym):
    """
    Get the layer number from the given acronym.

    Parameters:
        acronym (str): The acronym representing a cortical layer.

    Returns:
        layer_number (int): The layer number corresponding to the given acronym.
    """
    try:
        layer = int(re.findall(r'\d+', acronym)[0])
        if layer == 3:
            layer = 0
        return layer
    except IndexError:
        return 0
    
def get_structure_ids(df, annotations):
    """
    Get the structure IDs for the given DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame containing coordinate values.
        annotations (np.ndarray): The annotation volume.

    Returns:
        structure_ids (np.ndarray): The structure IDs corresponding to the given DataFrame.
    """
    x = (df.anterior_posterior_ccf_coordinate.values / 10).astype('int')
    y = (df.dorsal_ventral_ccf_coordinate.values / 10).astype('int')
    z = (df.left_right_ccf_coordinate.values / 10).astype('int')
    
    x[x < 0] = 0
    y[y < 0] = 0
    z[z < 0] = 0
    
    structure_ids = annotations[x, y, z] - 1 # annotation volume is Matlab-indexed
    
    return structure_ids

# %% First half of the layer assignment (called by the second half)

def cortical_depth_calculation(channels) -> pd.DataFrame:
    """
    Calculate the cortical depth for the given channels.

    Parameters:
        channels (pd.DataFrame): The DataFrame containing channel information.

    Returns:
        pd.DataFrame: The DataFrame with a new column 'cortical_depth' that represents the cortical depth of each channel.
    """
    channels = channels[channels.anterior_posterior_ccf_coordinate > 0]

    x = (channels.anterior_posterior_ccf_coordinate.values / 10).astype('int')
    y = (channels.dorsal_ventral_ccf_coordinate.values / 10).astype('int')
    z = (channels.left_right_ccf_coordinate.values / 10).astype('int')

    cortical_depth = streamlines[x, y, z]

    channels['cortical_depth'] = 0

    channels.loc[channels.anterior_posterior_ccf_coordinate > 0, 'cortical_depth'] = cortical_depth
    
    return channels


# %% Second half of the layer assignment (called by the user and calls the first half)

def layer_assignment_to_channels(channels) -> pd.DataFrame:
    """
    Assign cortical layers to the given channels.

    ATTENTION! Please keep in mind that these layer assignments are only estimates, and not definitive labels. 

    Parameters:
        channels (pd.DataFrame): The DataFrame containing channel information.

    Returns:
        pd.DataFrame: The DataFrame with a new column 'cortical_layer' that assigns each channel to a cortical layer.
    """
    channels = cortical_depth_calculation(channels)

    structure_ids = get_structure_ids(channels, annotations)
    structure_acronyms = structure_tree.loc[structure_ids].acronym

    layers = [get_layer_name(acronym) for acronym in structure_acronyms]
            
    channels['cortical_layer'] = layers
    
    return channels
# %%

def cortical_layer_assignment(channels, units) -> pd.DataFrame:
    """
    Assigns cortical layers to units based on their ecephys_channel_id.
    
    **ATTENTION! Please keep in mind that these layer assignments are only estimates, and not definitive labels**. 

    Args:
        channels (pd.DataFrame): The DataFrame containing channel information.
        units (pd.DataFrame): The DataFrame containing unit information.

    Returns:
        pd.DataFrame: The DataFrame with an additional 'layer' column containing the cortical layer assignments for each unit.
    """
    
    # Get the channels with the layer assignment
    channels = layer_assignment_to_channels(channels)
            
    # Get the channel ids for the units
    unit_channel_ids = units.peak_channel_id
    
    # Filter channel_ids to only those that are in the channels index
    filtered_channel_ids = unit_channel_ids[unit_channel_ids.isin(channels.index)]

    # Get the layer assignments for the filtered units
    layer_assignments = channels.loc[filtered_channel_ids].cortical_layer.values
    
    # Initialize a new column in units with a default value
    units['layer'] = np.nan

    # Update only the filtered units
    units.loc[units['peak_channel_id'].isin(filtered_channel_ids), 'layer'] = layer_assignments

    return units
