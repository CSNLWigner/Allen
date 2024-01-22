import csv
def save_dict_items(dictionary, name="", path="results"):
    if name != "":
        name = name + "_"
    if path[-1] != "/":
        path = path + "/"
    for key, value in dictionary.items():
        with open(f"{path}{name}{key}.csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([value])
