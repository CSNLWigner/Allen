import csv
import numpy as np

def save_dict_items(dictionary, name="", path="results"):
    if name != "":
        name = name + "_"
    if path[-1] != "/":
        path = path + "/"
    for key, value in dictionary.items():
        if type(value) == np.ndarray:
            np.savetxt(f"{path}{name}{key}.csv", delimiter=",")
        else:
            "pickle"
