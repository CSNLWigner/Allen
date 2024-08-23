# layers.py

"""
Module for layer analysis.

Functions:
- get_layers(units): Get the list of cortical layers from unit tables.
"""

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd


# Get the list of cortical layers from unit tables
def get_layers(units: pd.DataFrame) -> list:
    """
    Get unique layers from the given DataFrame.

    Parameters:
    units (pd.DataFrame): The DataFrame containing the units.

    Returns:
    list: A list of unique layers.

    """
    # Get the unique layers
    layers = units['ecephys_structure_acronym'].unique()
    
    # Sort the layers
    layers.sort()
    
    # Return the layers
    return layers
