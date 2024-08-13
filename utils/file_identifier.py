import argparse
import csv
import json
import h5py

def detect_file_format(file_path):
    try:
        with open(file_path, 'r') as f:
            first_line = f.readline().strip()

            # Check for JSON format
            try:
                json.loads(first_line)
                return 'JSON'
            except json.JSONDecodeError:
                pass

            # Check for CSV format
            try:
                csv.Sniffer().sniff(first_line)
                return 'CSV'
            except csv.Error:
                pass

            # Check for XML format
            if first_line.startswith('<?xml'):
                return 'XML'

    except UnicodeDecodeError:
        # If the file is not a text file, try to open it as an NWB file
        try:
            with h5py.File(file_path, 'r') as f:
                if 'nwb_version' in f.attrs:
                    return 'NWB'
        except Exception as e:
            print(f"Error opening file as NWB: {e}")
            pass

    return 'Unknown format'

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Detect the format of a file.')
    parser.add_argument('file_path', type=str, help='The path to the file to be checked.')

    # Parse the arguments
    args = parser.parse_args()

    # Use the parsed file path
    file_format = detect_file_format(args.file_path)
    print(f'The file format is: {file_format}')
