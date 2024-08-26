# analyses/layer_interaction_maxValues.py

"""
This module contains tools for finding the global maxima in a given data.

Functions:
- getUnitsNumber(areaName: str) -> dict: Get the number of units for each layer in the origin and target areas.
- find_max_value(data) -> tuple: Find the maximum value in a given data array.
- getGlobalMaxima(dataDict: dict, slice_index: slice) -> dict: Finds the global maxima in a given data dictionary for a specific slice.
- update_csv(data: pd.DataFrame, filename: str, force=False) -> None: Appends the given data to a CSV file if certain keys are not already present.
"""

import numpy as np
import yaml
from scipy.signal import argrelextrema

from utils.data_io import load_pickle

timeBin = yaml.safe_load(open("params.yaml"))["crosstime"]['scaling-factor']

areaDict = {
    'V1': 'VISp',
    'LM': 'VISl'
}

def getUnitsNumber(areaName: str) -> dict:
    """
    Get the number of units for each layer in the origin and target areas.

    Args:
        areaName (str): The name of the area.

    Returns:
        dict: A dictionary containing the number of units for each layer in the origin and target areas.
              The keys are the layer names.
    """
    
    # Load layer-assignments
    layerAssignment = load_pickle(f'layer-assignments-{areaDict[areaName]}', path='data/units')
    
    # Count the number of units for each layer in the origin and target areas
    unitNumber = {layer: len(layerAssignment[layerAssignment == layer]) for layer in layerAssignment.unique()}
    
    return unitNumber


def find_max_value(data):

    max_ind = np.unravel_index(
        np.nanargmax(data),  # argrelextrema(dataSlice, np.greater)
        data.shape)  # in case of nan error, preprocess the array by replacing nan with -np.inf

    # get the actual values using these indices
    max_val = data[max_ind]
    
    return max_val, max_ind

def getGlobalMaxima(dataDict: dict, slice_index: slice) -> dict:
    """
    Finds the global maxima in a given data dictionary for a specific slice.

    Args:
        dataDict (dict): A dictionary containing the data.
        slice_index (slice): The index of the slice to consider.

    Returns:
        dict: A dictionary containing the local maxima values and indices.
    """

    result_dict = {}

    for output in dataDict.keys():
        for input in dataDict[output].keys():
            dataSlice = dataDict[output][input][slice_index]

            # Check if all values in dataSlice are NaN
            if np.all(np.isnan(dataSlice)):
                max_val, max_ind = np.nan, None  # Placeholder for indices, adjust as needed
            else:
                max_val, max_ind = find_max_value(dataSlice)
            
            # turn the indices to timepoints by multiplying with the timeBin
            if type(max_ind) is tuple:
                max_ind = tuple([i * timeBin for i in max_ind])
            
            # get the maximum value and index
            result_dict[f"{output}-{input}-val"] = max_val
            result_dict[f"{output}-{input}-ind"] = max_ind
    
    return result_dict


import pandas as pd


def update_csv(data: pd.DataFrame, filename: str, force=False) -> None:
    """
    Appends the given data to a CSV file if certain keys are not already present.

    Args:
        data (pd.DataFrame): The data to be append to the end of the CSV file as a row.
        filename (str): The path of the CSV file.
        force (bool): If True, the existing row with the same fixedKeys values will be replaced. Default is False.

    Returns:
        None
    """
    
    # Fixed keys
    fixedKeys = ['session', 'direction', 'slice']

    # Step 1: Read the existing data
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        df = pd.DataFrame(columns=fixedKeys)

    # Ensure all fixedKeys are in the DataFrame, add them if not
    for key in fixedKeys:
        if key not in df.columns:
            df[key] = None  # or pd.NA or another appropriate default value

    # Step 2: Check for duplicates and append or replace data
    if not df.empty:
        # Create a mask for rows in 'df' that match the 'fixedKeys' in 'data'
        mask = pd.concat([df[fixedKeys] == data[fixedKeys].iloc[0]]).all(axis=1)
        if mask.any():
            if force:
                # If forcing, drop the matching rows and append 'data'
                df = df[~mask]
            else:
                # If not forcing, and there's a match, do not append 'data'
                return
    # Append 'data' if no duplicates or if forcing
    df = pd.concat([df, data], ignore_index=True)

    # Step 3: Save the updated DataFrame back to the CSV
    df.to_csv(filename, index=False)


