import csv
import pickle
import numpy as np
import os


def save_csv(data, name, path="results"):
    if not os.path.exists(path):
        os.makedirs(path)
    with open(f"{path}/{name}.csv", "w") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerows(data)


def save_pickle(data, name, path="results"):
    if not os.path.exists(path):
        os.makedirs(path)
    with open(f"{path}/{name}.pickle", "wb") as f:
        pickle.dump(data, f)
        
def save_based_on_type(data, name, path="results"):
    if type(data) == np.ndarray:
        save_csv(data, name, path=path)
    else:
        save_pickle(data, name, path=path)

def save_dict_items(dictionary, name="", path="results"):
    if name != "":
        name = name + "_"
    for key, value in dictionary.items():
        save_pickle(value, f"{name}{key}", path=path)

############################################################################################################

def load_csv(name):
    with open(f"{name}.csv", "r") as f:
        reader = csv.reader(f, delimiter=",")
        data = np.array(list(reader)) # float?
    return data

def load_pickle(name):
    with open(f"{name}.pickle", "rb") as f:
        data = pickle.load(f)
    return data