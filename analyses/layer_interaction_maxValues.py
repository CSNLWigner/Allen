
import numpy as np
import yaml
from scipy.signal import argrelextrema

from utils.data_io import load_pickle

timeBin = yaml.safe_load(open("params.yaml"))["crosstime"]['scaling-factor']

areaName = {
    'V1': 'VISp',
    'LM': 'VISl'
}

def getUnitsNumber(originArea: str, targetArea: str) -> dict:
    """
    Get the number of units for each layer in the origin and target areas.

    Args:
        originArea (str): The name of the origin area.
        targetArea (str): The name of the target area.

    Returns:
        dict: A dictionary containing the number of units for each layer in the origin and target areas.
              The keys are in the format '{originArea}-{layer}-units' and '{targetArea}-{layer}-units'.
    """
    
    # Load layer-assignments
    layer_assignments_originArea = load_pickle(f'layer-assignments-{areaName[originArea]}', path='data/units')
    layer_assignments_targetArea = load_pickle(f'layer-assignments-{areaName[targetArea]}', path='data/units')
    
    # Count the number of units for each layer in the origin and target areas
    n_units_originArea = {layer: len(layer_assignments_originArea[layer_assignments_originArea == layer]) for layer in layer_assignments_originArea.unique()}
    n_units_targetArea = {layer: len(layer_assignments_targetArea[layer_assignments_targetArea == layer]) for layer in layer_assignments_targetArea.unique()}
    
    return {f'{originArea}-{layer}-units': n_units_originArea[layer] for layer in n_units_originArea.keys()} | {f'{targetArea}-{layer}-units': n_units_targetArea[layer] for layer in n_units_targetArea.keys()}


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


def update_csv(data: dict, filename: str, force=False) -> None:
    """
    Appends the given data to a CSV file if certain keys are not already present.

    Args:
        data (dict): The data to be append to the end of the CSV file as a row.
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
        df = pd.DataFrame(columns=data.keys())

    # Ensure all fixedKeys are in the DataFrame, add them if not
    for key in fixedKeys:
        if key not in df.columns:
            df[key] = None  # or pd.NA or another appropriate default value
    
    # Step 2: Check if the row with the same fixedKeys values exists
    if df[(df['session'] == data['session']) & (df['direction'] == data['direction']) & (df['slice'] == data['slice'])].empty:
        # Append the new data
        df = df.append(data, ignore_index=True)
    elif force:
        df = df.drop(df[(df['session'] == data['session']) & (df['direction'] == data['direction']) & (df['slice'] == data['slice'])].index)
        df = df.append(data, ignore_index=True)

    # Step 3: Overwrite the whole file
    df.to_csv(filename, index=False)


