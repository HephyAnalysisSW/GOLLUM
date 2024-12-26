import sys, os
sys.path.insert( 0, "..")
sys.path.insert( 0, "../..")

import common.user   as user
import common.data_structure as data_structure
from   common.selections import selections

from data_loader.data_loader_2 import H5DataLoader

def generate_filename(process=None, values=None):
    """
    Generates filenames based on a process and systematic values.

    Args:
    - process (str): The process name, one of 'diboson', 'ttbar', 'htautau', 'ztautau', or None.
    - values (tuple): A tuple of three numerical values for 'tes', 'jes', and 'met', in that order.

    Returns:
    - str: The generated filename.
    """

    # Handle the case when we want the nominal
    if values == None:
        values = data_structure.default_values

    # Create the formatted systematic parts
    sys_parts = []
    for sys_name, default, value in zip(data_structure.systematics, data_structure.default_values, values):
        if value != default:
            sys_parts.append(f"{sys_name}_{str(value).replace('.', 'p')}")

    # Combine all parts
    filename_parts = []
    if process:
        filename_parts.append(process)
    
    if sys_parts:
        filename_parts.extend(sys_parts)
    elif process:  # Add '_nominal' for process when no systematics deviate
        filename_parts.append("nominal")
    else:  # Completely nominal case with no process
        filename_parts.append("nominal")

    return "_".join(filename_parts) + ".h5"

import re

def parse_filename(filename):
    """
    Parses a filename to extract the process and systematic values.

    Args:
    - filename (str): The filename to parse, e.g., 'diboson_tes_3p99_jes_1p01_met_3.h5'.

    Returns:
    - tuple: (process, values), where
        - process (str or None): The process name or None for nominal cases.
        - values (tuple): A tuple of three numerical values for 'tes', 'jes', and 'met' in that order.
    """
    #FIXME: Remove this line: process_pattern = r"^(?!tes|jes|met)(.*?)_"  # Matches the process name at the start, avoiding 'tes', 'jes', 'met'
    # Regex to match process and systematics
    process_pattern = r"^(?!tes|jes|met)(.*?)(?:_|$)" # Matches the process name at the start, avoiding 'tes', 'jes', 'met'


    systematic_pattern = r"(tes|jes|met)_(\d+p\d+|\d+)"  # Matches systematics like 'tes_3p99', 'jes_1p01'

    # Remove the '.h5' extension
    base = filename.removesuffix(".h5")  # Fixed from rstrip

    # Extract process (if present)
    process_match = re.match(process_pattern, base)
    if process_match:
        process = process_match.group(1)
        base = base[len(process) + 1:]  # Remove process from the base string
    else:
        process = None

    # Extract systematics
    values = [1, 1, 0]  # Default values for tes, jes, met
    for sys_match in re.finditer(systematic_pattern, base):
        sys_name, value_str = sys_match.groups()

        # Explicitly handle "p" in systematic values
        if "p" in value_str:
            parts = value_str.split("p")
            value = float(parts[0]) + float(f"0.{parts[1]}")
        else:
            value = float(value_str)

        if sys_name == "tes":
            values[0] = value
        elif sys_name == "jes":
            values[1] = value
        elif sys_name == "met":
            values[2] = value

    return process, tuple(values)

#examples = [
#    ('diboson', (1, 0.97, 0)),
#    ('diboson', (1, 0.98, 0)),
#    ('diboson', (1, 0.99, 0)),
#    ('diboson', (1, 1.01, 0)),
#    ('diboson', (1, 1.02, 0)),
#    ('diboson', (1, 1.03, 0)),
#    ('diboson', (1, 1, 1.5)),
#    ('diboson', (1, 1, 3)),
#    ('diboson', (1, 1, 4.5)),
#    ('diboson', (1, 1, 6)),
#    ('diboson', (1, 1, 0)),
#    ('diboson', (0.97, 1, 0)),
#    ('diboson', (0.98, 1, 0)),
#    ('diboson', (0.99, 1, 0)),
#    ('diboson', (1.01, 1, 0)),
#    ('diboson', (1.02, 1, 0)),
#    ('diboson', (1.03, 1, 0)),
#    ('htautau', (1, 0.97, 0)),
#    ('htautau', (1, 0.98, 0)),
#    ('htautau', (1, 0.99, 0)),
#    ('htautau', (1, 1.01, 0)),
#    ('htautau', (1, 1.02, 0)),
#    ('htautau', (1, 1.03, 0)),
#    ('htautau', (1, 1, 1.5)),
#    ('htautau', (1, 1, 3)),
#    ('htautau', (1, 1, 4.5)),
#    ('htautau', (1, 1, 6)),
#    ('htautau', (1, 1, 0)),
#    ('htautau', (0.97, 1, 0)),
#    ('htautau', (0.98, 1, 0)),
#    ('htautau', (0.99, 1, 0)),
#    ('htautau', (1.01, 1, 0)),
#    ('htautau', (1.02, 1, 0)),
#    ('htautau', (1.03, 1, 0)),
#    ('ztautau', (1, 1, 1.5)),
#    ('ztautau', (1, 1, 3)),
#    ('ztautau', (1, 1, 4.5)),
#    ('ztautau', (1, 1, 6)),
#    ('ztautau', (1, 0.97, 0)),
#    ('ztautau', (1, 0.98, 0)),
#    ('ztautau', (1, 0.99, 0)),
#    ('ztautau', (1, 1.01, 0)),
#    ('ztautau', (1, 1.02, 0)),
#    ('ztautau', (1, 1.03, 0)),
#    ('ztautau', (0.97, 1, 0)),
#    ('ztautau', (0.98, 1, 0)),
#    ('ztautau', (0.99, 1, 0)),
#    ('ztautau', (1.01, 1, 0)),
#    ('ztautau', (1.02, 1, 0)),
#    ('ztautau', (1.03, 1, 0)),
#    ('ttbar', (1, 0.97, 0)),
#    ('ttbar', (1, 0.98, 0)),
#    ('ttbar', (1, 0.99, 0)),
#    ('ttbar', (1, 1.01, 0)),
#    ('ttbar', (1, 1.02, 0)),
#    ('ttbar', (1, 1.03, 0)),
#    ('ttbar', (1, 1, 1.5)),
#    ('ttbar', (1, 1, 3)),
#    ('ttbar', (1, 1, 4.5)),
#    ('ttbar', (1, 1, 6)),
#    ('ttbar', (1, 1, 0)),
#    ('ttbar', (0.97, 1, 0)),
#    ('ttbar', (0.98, 1, 0)),
#    ('ttbar', (0.99, 1, 0)),
#    ('ttbar', (1.01, 1, 0)),
#    ('ttbar', (1.02, 1, 0)),
#    ('ttbar', (1.03, 1, 0)),
#    (None, (0.97, 1, 0)),
#    (None, (0.98, 1, 0)),
#    (None, (0.99, 1, 0)),
#    (None, (1.01, 1, 0)),
#    (None, (1.02, 1, 0)),
#    (None, (1.03, 1, 0)),
#    (None, (1, 0.97, 0)),
#    (None, (1, 0.98, 0)),
#    (None, (1, 0.99, 0)),
#    (None, (1, 1.01, 0)),
#    (None, (1, 1.02, 0)),
#    (None, (1, 1.03, 0)),
#    (None, (1, 1, 1.5)),
#    (None, (1, 1, 3)),
#    (None, (1, 1, 4.5)),
#    (None, (1, 1, 6)),
#    (None, (1, 1, 0)),
#]
#
## Apply the generate_filename function to all examples
#for process, values in examples:
#    print(generate_filename(process, values))
#    print(os.path.exists( os.path.join( '/eos/vbc/group/mlearning/data/Higgs_uncertainty/input_data/split_train_dataset/', generate_filename(process, values))) )

class Dataset:

    def __init__( self, selection_dir="inclusive", selection_function=None, systematic=None, process=None, data_directory=user.derived_data_directory):


        #if selection_dir == "inclusive":
        #    self.data_directory = user.data_directory
        #    self.subdir=''
        #else:
        #    self.data_directory = user.derived_data_directory
        #    self.subdir=selection_dir

        self.data_directory = data_directory
        self.selection_dir  = selection_dir
        self.file_name = generate_filename(process, systematic)
        self.file_path = os.path.join(self.data_directory, self.selection_dir, self.file_name)
        
        self.selection_function = selection_function

        if not os.path.exists( self.file_path ):
            raise RuntimeError(f"File {self.file_path} not found!")

        print(f"Loading from {self.file_path}")

    def get_data_loader( self, n_split = 100, batch_size=None, selection_function=None):
        
        if selection_function is not None:
            sf = selection_function
        else:
            sf = self.selection_function 

        return H5DataLoader(
            file_path = self.file_path,
            batch_size= batch_size,
            n_split   = n_split,
            selection_function = sf
            )
if __name__=="__main__":
    d0 = Dataset(selection_function=selections['lowMT_VBFJet'])
    d1 = Dataset(selection_dir='lowMT_VBFJet')
