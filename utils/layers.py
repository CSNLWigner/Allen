# layers.py

"""
Tools for simple layer analysis.

Functions:
- get_layers(units): Get the list of cortical layers from unit tables.
"""

from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

def get_layers(units: pd.DataFrame) -> list:
    """
    Get the unique cortical layers from the given DataFrame.

    Parameters:
    units (pd.DataFrame): The DataFrame containing the units.

    Returns:
    list: A list of unique cortical layers.

    """
    # Get the unique layers
    layers = units['ecephys_structure_acronym'].unique()
    
    # Sort the layers
    layers.sort()
    
    # Return the layers
    return layers
