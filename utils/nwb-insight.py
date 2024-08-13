from pynwb import NWBHDF5IO, validate

# Open the NWB file
file_path = 'data/.from_warehouse/1026122596.nwb'
io = NWBHDF5IO(file_path, 'r')

# Validate the file
validation_errors = validate(io)

# Validate the file 2/2
if validation_errors:
    print("The file is not an NWB file.")
    exit()
else:
    print("The file is an NWB file.")

# Read the NWB file
nwbfile = io.read()

# Print some important metadata
print("NWB File Metadata:")
print(f"Session Description: {nwbfile.session_description}")
print(f"Identifier: {nwbfile.identifier}")
print(f"Session Start Time: {nwbfile.session_start_time}")
print(f"Experimenter: {nwbfile.experimenter}")

# Close the file
io.close()
