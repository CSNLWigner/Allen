
import csv

import numpy as np
from scipy.signal import argrelextrema

from utils.data_io import load_pickle

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

def getLocalMaxima(dataDict: dict, slice_index: slice) -> dict:
    """
    Finds the local maxima in a given data dictionary for a specific slice.

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

            # determine the indices of the local maxima
            # in case of nan error, preprocess the array by replacing nan with -np.inf
            max_ind = np.unravel_index(
                         argrelextrema(dataSlice, np.greater), 
                         dataSlice.shape)

            # get the actual values using these indices
            max_val = dataSlice[max_ind]
            
            # get the maximum value and index
            result_dict[f"{output}-{input}-val"] = max_val
            result_dict[f"{output}-{input}-ind"] = max_ind
    
    return result_dict


def update_csv(data: dict, filename: str) -> None:
    """
    Appends the given data to a CSV file if certain keys are not already present.

    Args:
        data (dict): The data to be written to the CSV file.
        filename (str): The path of the CSV file.

    Returns:
        None
    """
    # Fixed keys
    fixedKeys = ['session', 'direction', 'slice']
    should_append = False

    # Open the file in read mode to check conditions
    try:
        with open(filename, 'r', newline='') as file:
            # Check if the fixedKeys are already present in the table
            if not any(all(row[key] == data[key] for key in fixedKeys) for row in csv.DictReader(file)):
                should_append = True
    except FileNotFoundError:
        # If the file doesn't exist, we'll create it, so we should append.
        should_append = True

    # If conditions are met, open the file in append mode to write data
    if should_append:
        with open(filename, 'a', newline='') as file:
            dictWriter = csv.DictWriter(file, fieldnames=data.keys())
            dictWriter.writerow(data)


