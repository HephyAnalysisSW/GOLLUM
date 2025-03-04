from sklearn.isotonic import IsotonicRegression
import pickle
import sys
import os
import yaml
from tqdm import tqdm
sys.path.insert( 0, '..')
sys.path.insert( 0, '../..')

from common.logger import get_logger

import numpy as np

import common.user as user
import common.selections as selections
import common.datasets_hephy as datasets_hephy

## Iterate through the dataset
#loader = datasets_hephy.get_data_loader(selection="lowMT_VBFJet", n_split=1)
#for batch in loader:
#    data, weights, labels = loader.split(batch)
#    print(data.shape, weights.shape, labels.shape, np.unique(labels, return_counts=True) )
#
#    print(" class probabilities from TFMC")
#    prob = tfmc.predict(data, ic_scaling = False)
#    print(prob)
#
#    break

class Calibration:

    def __init__( self, yaml_config=None, selection=None, n_split=10, small=False):

        self.selection   = selection
        self.n_split     = n_split if not small else 100
        self.small = small

        if yaml_config is not None:
            with open(yaml_config) as f:
                cfg_ = yaml.safe_load(f)
                self.cfg = cfg_['MultiClassifier'][selection]

            logger.info("Config loaded from Workflow/configs/{}".format(args.config))

            if self.cfg['module']=='TFMC':
                from ML.TFMC.TFMC import TFMC
                self.classifier = TFMC.load(self.cfg['model_path'])
            elif self.cfg['module']=='XGBMC':
                from ML.XGBMC.XGBMC import XGBMC
                self.classifier = XGBMC.load(self.cfg['model_path'])
            else:
                raise NotImplementedError

    def train( self ):
        logger.info(f"Training: Load data for {self.selection}")

        self.loader = datasets_hephy.get_data_loader(selection=self.selection, n_split=self.n_split)

        # Initialize lists for accumulation
        all_prob = []
        all_weights = []
        all_labels = []

        for batch in tqdm(self.loader, desc="Processing batches"):
            data, weights, labels = self.loader.split(batch)  # Unpack batch

            prob = self.classifier.predict(data, ic_scaling=True)  # Predict with ic_scaling
            all_prob.append(prob)
            all_weights.append(weights)
            all_labels.append(labels)

            del data  # Explicitly delete data to free memory
            if self.small: break

        # Concatenate into final NumPy arrays
        all_prob = np.concatenate(all_prob, axis=0)
        all_weights = np.concatenate(all_weights, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        all_prob/=all_prob.sum(axis=1, keepdims=True)

        truth       = (all_labels == 0).astype(float) # get truth label
        logger.info( "Training IsotonicRegression." ) 
        self.iso_reg = IsotonicRegression(out_of_bounds='clip', y_min=1e-6, y_max=1.-1e-6).fit(all_prob[:, 0], truth, sample_weight=all_weights)
        logger.info( "Done." ) 
        
    def save( self, file_name ):
        with open(filename, 'wb') as file:
            pickle.dump(self.iso_reg, file)

        logger.info(f"Written {filename}")

    @classmethod
    def load( cls, file_name ):
        new_instance = cls()
        with open(filename, 'rb') as file:
            new_instance.iso_reg = pickle.load(file)

        logger.info(f"Loaded Calibration {filename}")

if __name__=="__main__":
    import argparse
    # Argument parser setup
    parser = argparse.ArgumentParser(description="ML inference.")
    parser.add_argument('--logLevel', action='store', nargs='?', choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'TRACE', 'NOTSET'], default='ERROR', help="Log level for logging")
    parser.add_argument("--config", default = "config_reference_v2_sr", help="Path to the config file.")
    parser.add_argument("--selection", default="lowMT_VBFJet", help="Which selection?")
    parser.add_argument("--save", action="store_true", help="Save the ML predictions for the simulation.")
    parser.add_argument("--small", action="store_true", help="Run a subset.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    parser.add_argument("--postfix", default = None, type=str,  help="Append this to the fit result.")

    args = parser.parse_args()

    logger  = get_logger(args.logLevel, logFile = None)

    subdirs = [args.selection]

    # Where to store the training
    model_directory = os.path.join(user.model_directory, "Calibration", *subdirs,  args.config, args.selection)
    os.makedirs(model_directory, exist_ok=True)
    filename = os.path.join( model_directory, 'calibrator.pkl')

    if os.path.exists( filename ) and not args.overwrite:
        logger.info(f"Found {filename}. Do nothing")
        sys.exit(0)
    elif os.path.exists( filename ) and args.overwrite:
        logger.warning(f"Will overwrite {filename}")    

    calib = Calibration( os.path.join( os.path.abspath('../../Workflow/configs'), args.config) +".yaml", selection=args.selection, small=args.small)

    calib.train()
    
    calib.save(model_directory)

