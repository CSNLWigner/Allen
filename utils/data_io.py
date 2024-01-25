import csv
import pickle
import numpy as np

def save_csv(data, name):
    with open(f"{name}data.csv", "w") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerows(data)
        
def save_pickle(data, name):
    with open(f"{name}data.pickle", "wb") as f:
        pickle.dump(data, f)
        
def save_based_on_type(data, name):
    if type(data) == np.ndarray:
        np.savetxt(f"{name}data.csv", data, delimiter=",")
    else:
        save_pickle(data, name)

def save_dict_items(dictionary, name="", path="results"):
    if name != "":
        name = name + "_"
    if path[-1] != "/":
        path = path + "/"
    for key, value in dictionary.items():
        save_based_on_type(value, f"{path}{name}{key}")

############################################################################################################

def load_csv(name):
    with open(f"{name}data.csv", "r") as f:
        reader = csv.reader(f, delimiter=",")
        data = np.array(list(reader)) # float?
    return data

def load_pickle(name):
    with open(f"{name}data.pickle", "rb") as f:
        data = pickle.load(f)
    return data