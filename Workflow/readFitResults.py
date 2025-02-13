import pickle
import argparse

parser = argparse.ArgumentParser(description="ML inference.")
parser.add_argument('--file', action='store', default=None)
args = parser.parse_args()

with open(args.file, 'rb') as file:
    fitResult = pickle.load(file)

for key in fitResult.keys():
    print(key,":", fitResult[key])
