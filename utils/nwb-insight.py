from IPython.display import display
from nwbwidgets import nwb2widget
from pynwb import NWBHDF5IO

# Open the NWB file
file_path = 'path_to_your_file.nwb'
io = NWBHDF5IO(file_path, 'r')
nwbfile = io.read()

# Print some important metadata
print("NWB File Metadata:")
print(f"Session Description: {nwbfile.session_description}")
print(f"Identifier: {nwbfile.identifier}")
print(f"Session Start Time: {nwbfile.session_start_time}")
print(f"Experimenter: {nwbfile.experimenter}")
print(f"Lab: {nwbfile.lab}")
print(f"Institution: {nwbfile.institution}")

# Explore the acquisition data
print("\nAcquisition Data:")
acquisition_data = nwbfile.acquisition
for name, data in acquisition_data.items():
    print(name, data)

# Visualize the data
widget = nwb2widget(nwbfile)
display(widget)

# Close the file
io.close()
