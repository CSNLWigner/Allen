import argparse
from pynwb import NWBHDF5IO, validate

def main(file_path):
    # Open the NWB file
    io = NWBHDF5IO(file_path, 'r')

    # Validate the NWB file
    validation_errors = validate(io)
    if validation_errors:
        print("The file is not a valid NWB file.")
        print("Validation errors:")
        for error in validation_errors:
            print(f"- {error}")
        io.close()
        return

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

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Inspect an NWB file.')
    parser.add_argument('file_path', type=str, help='The path to the NWB file to be inspected.')

    # Parse the arguments
    args = parser.parse_args()

    # Use the parsed file path
    main(args.file_path)
