
import yaml

from analyses.layer_interaction_maxValues import getLocalMaxima, getUnitsNumber, update_csv
from utils.data_io import load_pickle

session = yaml.safe_load(open("params.yaml"))["load"]['session']
crosstime = yaml.safe_load(open("params.yaml"))["crosstime"]

# for direction in ['bottom-up', 'top-down']:
for originArea, targetArea in zip(['V1', 'LM'], ['LM', 'V1']):
    print(f"Plotting {originArea} to {targetArea}")
    
    # Results
    data = load_pickle(f"layer-interaction_{originArea}-to-{targetArea}") # result has a shape of (timepoints X timepoints) for each layer-pair (output-input)
    
    # Areas Of Interest
    AOI = {
        'first': slice(0, int(100/crosstime['scaling-factor'])),
        'second':   slice(int(100/crosstime['scaling-factor']), int(200/crosstime['scaling-factor']))
    }

    # Calculate the maximum values for each layer-pair
    for slice_name, slice_index in AOI.items():

        # Get the maximum values for each layer-pair
        maxValues = getLocalMaxima(data, slice_index)

        # Update the maxValues dictionary with the fixed keys
        maxValues['session'] = session
        maxValues['direction'] = f"{originArea}-to-{targetArea}"
        maxValues['slice'] = slice_name

        # Update the maxValues dictionary with the number of units in the origin and target areas
        maxValues.update(getUnitsNumber(originArea, targetArea))

        # Update the CSV file
        update_csv(maxValues, f"results/maxValues.csv")
