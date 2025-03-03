import os
import yaml
import pickle

def convert_yaml_to_pickle(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".yaml") or filename.endswith(".yml"):
            yaml_path = os.path.join(directory, filename)
            pkl_path = os.path.join(directory, os.path.splitext(filename)[0] + ".pkl")
            
            with open(yaml_path, 'r', encoding='utf-8') as yaml_file:
                data = yaml.safe_load(yaml_file)
            
            with open(pkl_path, 'wb') as pkl_file:
                pickle.dump(data, pkl_file)
            
            print(f"Converted: {yaml_path} -> {pkl_path}")

if __name__ == "__main__":
    directory = "."  # Change this to the desired directory
    convert_yaml_to_pickle(directory)

