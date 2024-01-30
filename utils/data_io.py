import csv
import pickle
import numpy as np
import os
import matplotlib.figure


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

def save_fig(fig:matplotlib.figure.Figure, name, path="figures"):
    if not os.path.exists(path):
        os.makedirs(path)
    fig.savefig(f"{path}/{name}.png")
        
def save_based_on_type(data, name, path="results"):
    if type(data) == np.ndarray:
        save_csv(data, name, path=path)
    else:
        save_pickle(data, name, path=path)

def save_dict_items(dictionary, name="", path="results", log=True):
    if name != "":
        name = name + "_"
    for key, value in dictionary.items():
        if log:
            print("Saving", key, type(value))
        save_pickle(value, f"{name}{key}", path=path)

############################################################################################################

def load_csv(name, path=""):
    if path != "":
        name = f'{path}/{name}'
    with open(f"{name}.csv", "r") as f:
        reader = csv.reader(f, delimiter=",")
        data = np.array(list(reader)) # float?
    return data

def load_pickle(name, path=""):
    if path != "":
        name = f'{path}/{name}'
    with open(f"{name}.pickle", "rb") as f:
        data = pickle.load(f)
    return data