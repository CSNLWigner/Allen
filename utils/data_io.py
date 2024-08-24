# utils/data_io.py

"""
This submodule contains tools for saving and loading data.

Functions:
- path_name(path: str, name: str) -> str: Returns the full path name.
- save_csv(data: List[List[Any]], name: str, path: str = "results") -> None: Saves the data to a CSV file.
- save_pickle(data: Any, name: str, path: str = "results") -> None: Saves the data to a pickle file.
- save_fig(fig: matplotlib.figure.Figure, name: str, path: str = "figures") -> None: Saves the figure to a PNG file.
- save_based_on_type(data: Any, name: str, path: str = "results") -> None: Saves the data based on its type.
- save_dict_items(dictionary: Dict[str, Any], name: str = "", path: str = "results", log: bool = True) -> None: Saves the items in a dictionary.
- load_csv(name: str, path: str = "") -> pd.DataFrame: Loads the data from a CSV file.
- load_pickle(name: str, path: str = "results") -> Any: Loads the data from a pickle file.
"""

import csv
import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Any

import matplotlib.figure

def path_name(path: str, name: str) -> str:
    """
    Returns the full path name by concatenating the path and name.

    Args:
        path (str): The path to the file.
        name (str): The name of the file.

    Returns:
        str: The full path name.
    """
    if path != "":
        if path[-1] != "/":
            path = path + "/"
        name = f'{path}{name}'
    return name


def save_csv(data: List[List[Any]], name: str, path: str = "results") -> None:
    """
    Saves the data to a CSV file.

    Args:
        data (List[List[Any]]): The data to be saved.
        name (str): The name of the file.
        path (str, optional): The path to save the file. Defaults to "results".
    """
    if not os.path.exists(path):
        os.makedirs(path)
    name = path_name(path, name)
    with open(f"{name}.csv", "w") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerows(data)


def save_pickle(data: Any, name: str, path: str = "results") -> None:
    """
    Saves the data to a pickle file.

    Args:
        data (Any): The data to be saved.
        name (str): The name of the file.
        path (str, optional): The path to save the file. Defaults to "results".
    """
    if not os.path.exists(path):
        os.makedirs(path)
    name = path_name(path, name)
    with open(f"{name}.pickle", "wb") as f:
        pickle.dump(data, f)


def save_fig(fig: matplotlib.figure.Figure, name: str, path: str = "figures") -> None:
    """
    Saves the figure to a PNG file.

    Args:
        fig (matplotlib.figure.Figure): The figure to be saved.
        name (str): The name of the file.
        path (str, optional): The path to save the file. Defaults to "figures".
    """
    if not os.path.exists(path):
        os.makedirs(path)
    fig.savefig(f"{path}/{name}.png")


def save_based_on_type(data: Any, name: str, path: str = "results") -> None:
    """
    Saves the data based on its type.

    Args:
        data (Any): The data to be saved.
        name (str): The name of the file.
        path (str, optional): The path to save the file. Defaults to "results".
    """
    if type(data) == np.ndarray:
        save_csv(data, name, path=path)
    else:
        save_pickle(data, name, path=path)


def save_dict_items(dictionary: Dict[str, Any], name: str = "", path: str = "results", log: bool = True) -> None:
    """
    Saves the items in a dictionary.

    Args:
        dictionary (Dict[str, Any]): The dictionary containing the items to be saved.
        name (str, optional): The name prefix for the saved files. Defaults to "".
        path (str, optional): The path to save the files. Defaults to "results".
        log (bool, optional): Whether to log the saving process. Defaults to True.
    """
    if name != "":
        name = name + "_"
    for key, value in dictionary.items():
        if log:
            print("Saving", key, type(value))
        save_pickle(value, f"{name}{key}", path=path)


def load_csv(name: str, path: str = "") -> pd.DataFrame:
    """
    Loads the data from a CSV file.

    Args:
        name (str): The name of the file.
        path (str, optional): The path to the file. Defaults to "".

    Returns:
        pd.DataFrame: The loaded data.
    """
    name = path_name(path, name)
    data = pd.read_csv(f"{name}.csv")
    return data


def load_pickle(name: str, path: str = "results") -> Any:
    """
    Loads the data from a pickle file.

    Args:
        name (str): The name of the file.
        path (str, optional): The path to the file. Defaults to "results".

    Returns:
        Any: The loaded data.
    """
    name = path_name(path, name)
    with open(f"{name}.pickle", "rb") as f:
        data = pickle.load(f)
    return data