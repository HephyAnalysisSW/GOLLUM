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
import matplotlib.pyplot as plt

import common.user as user
import common.selections as selections
import common.datasets as datasets

## Iterate through the dataset
#loader = datasets.get_data_loader(selection="lowMT_VBFJet", n_split=1)
#for batch in loader:
#    data, weights, labels = loader.split(batch)
#    print(data.shape, weights.shape, labels.shape, np.unique(labels, return_counts=True) )
#
#    print(" class probabilities from TFMC")
#    prob = tfmc.predict(data, ic_scaling = False)
#    print(prob)
#
#    break

class MultiClassCalibration:

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

        self.loader = datasets.get_data_loader(selection=self.selection, n_split=self.n_split)

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

        self.iso_reg = []
        for class_id in range(4):
            truth       = (all_labels == class_id).astype(float) # get truth label
            logger.info( f"Training IsotonicRegression for class {class_id}." ) 
            self.iso_reg.append(IsotonicRegression(out_of_bounds='clip', y_min=1e-6, y_max=1.-1e-6).fit(all_prob[:, class_id], truth, sample_weight=all_weights))
            logger.info( f"Isotonic Regressions of class {class_id} done." ) 
        self.data_for_plot = {"prob": all_prob, "weight": all_weights, "label": all_labels}
        logger.info( "All Isotonic Regressions Done." ) 
        
    def save( self, file_name ):
        with open(file_name, 'wb') as file:
            pickle.dump(self.iso_reg, file)

        logger.info(f"Written {file_name}")

    def load( self, file_name ):
        with open(file_name, 'rb') as file:
            self.iso_reg = pickle.load(file)
        #logger.info(f"Loaded Calibration {file_name}") #breaks loading from inference.py


    def predict(self, input_dcr):
        # assume for now that calibrator was trained / loaded
        # TO-DO: check if calibrator exists and train/load otherwise

        calibrated_dcr = np.zeros_like(input_dcr)
        for class_id in range(4):
            calibrated_dcr[:, class_id] = self.iso_reg[class_id].predict(input_dcr[:,class_id])
            # rescale each class independly based on the independenly trained calibrator

        # keep 0 (signal) at its value and rescale the other 3 such that the 4 sum to 1. 
        calibrated_dcr[:, 1:] = calibrated_dcr[:, 1:] * ((1.-calibrated_dcr[:, 0])/(calibrated_dcr[:, 1:].sum(axis=-1))).reshape(-1,1)
        
        return calibrated_dcr

    def plot_calibration(self, file_name):
        logger.info(f"started plotting calibration")
        fig, axs = plt.subplots(2, 2)
        for idx, loc in enumerate([(0,0), (0,1), (1,0), (1,1)]):
            uncalib_pred = self.data_for_plot['prob'][:, idx]
            truth = (self.data_for_plot['label'] == idx).astype(float)
            calib_pred = self.predict(self.data_for_plot['prob'])[:, idx]
            prob_true_uncalib, prob_pred_uncalib = weighted_calibration(truth, 
                                                                        uncalib_pred, 
                                                                        nbins=10, 
                                                                        weights = self.data_for_plot['weight'])
            prob_true_calib, prob_pred_calib = weighted_calibration(truth, 
                                                                    calib_pred, 
                                                                    nbins=10, 
                                                                    weights = self.data_for_plot['weight'])
            mask_uncalib = (prob_pred_uncalib == 0.) & (prob_true_uncalib == 0.)
            mask_calib = (prob_pred_calib == 0.) & (prob_true_calib == 0.)


            axs[loc].plot(prob_pred_uncalib[~mask_uncalib], prob_true_uncalib[~mask_uncalib], 
                          label="before calibration")
            axs[loc].plot(prob_pred_calib[~mask_calib], prob_true_calib[~mask_calib], 
                          label="after calibration")
            axs[loc].plot([0.,1.], [0.,1.], ls='dashed',c='k', label='identity')
            axs[loc].set_title(f"class {idx} calibration")
            axx = axs[loc].inset_axes([0.625, 0.175, 0.3, 0.3])
            axx.plot(prob_pred_uncalib[~mask_uncalib], prob_true_uncalib[~mask_uncalib], 
                          label="before calibration")
            axx.plot(prob_pred_calib[~mask_calib], prob_true_calib[~mask_calib], 
                          label="after calibration")

            axx.plot([0.,.2], [0.,.2], ls='dashed',c='k', label='identity')
            axx.set(xlim=(0,0.2), ylim=(0,0.2))
            
            if idx == 0:
                axs[loc].legend(loc='upper left')
    
        for ax in axs.flat:
            ax.set(xlabel='predicted DCR', ylabel='true DCR', xlim=(-0.05,1.05), ylim=(-0.05,1.05))
            ax.label_outer()
        
        plt.savefig(file_name)
        plt.close()
        logger.info(f"Saved calibration plot to {file_name}")

    def plot_IsotonicRegression(self, file_name):
        logger.info(f"Started isotonic regression plot")
        fig, axs = plt.subplots(2, 2)
        for idx, loc in enumerate([(0,0), (0,1), (1,0), (1,1)]):
            x_scan = np.linspace(0., 1., 1001)
            axs[loc].plot(x_scan, self.iso_reg[idx].predict(x_scan))
            axs[loc].plot([0.,1.], [0.,1.], ls='dashed',c='k', label='identity')
            axs[loc].set_title(f"class {idx} calibration")
    
        for ax in axs.flat:
            ax.set(xlabel='dcr_input', ylabel='dcr_output', xlim=(0,1), ylim=(0,1))
            ax.label_outer()
    
        for idx, loc in enumerate([(0,0), (0,1), (1,0), (1,1)]):
            axx = axs[loc].inset_axes([0.625, 0.175, 0.3, 0.3])
            x_scan = np.linspace(0., .2, 201)
            axx.plot(x_scan, self.iso_reg[idx].predict(x_scan))
            axx.plot([0.,.2], [0.,.2], ls='dashed',c='k', label='identity')
            axx.set(xlim=(0,0.2), ylim=(0,0.2))
        plt.savefig(file_name)
        plt.close()
        logger.info(f"Saved isotonic regression plot to {file_name}")


def weighted_calibration(true_label, pred_output, nbins=10, weights=None):
    # reproduces to a large extend 
    # from sklearn.calibration import calibration_curve
    # but adds the functionality to have weighted events
    bins = np.linspace(pred_output.min(), pred_output.max(), nbins+1)
    if weights is None:
        weights = np.ones_like(true_label)
    mask = (pred_output.reshape(-1,1) >= bins[:-1]) & (bins[1:] >= pred_output.reshape(-1,1))
    true_fraction = []
    pred_fraction = []
    for bin_nr in range(nbins):
        if len(weights[mask[:,bin_nr]]) == 0:
            true_fraction.append(0.)
        else:
            true_fraction.append((weights[mask[:,bin_nr]]*true_label[mask[:,bin_nr]]).sum() /(weights[mask[:,bin_nr]]).sum()+1e-16)
        if len(pred_output[mask[:,bin_nr]]) == 0:
            pred_fraction.append(0.)
        else:
            pred_fraction.append((pred_output[mask[:,bin_nr]]).mean())
    prob_true = np.array(true_fraction)
    prob_pred = np.array(pred_fraction)
    return prob_true, prob_pred        

if __name__=="__main__":
    import argparse
    # Argument parser setup
    parser = argparse.ArgumentParser(description="ML inference.")
    parser.add_argument('--logLevel', action='store', nargs='?', choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'TRACE', 'NOTSET'], default='INFO', help="Log level for logging")
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
    filename = os.path.join( model_directory, f'calibrator_multi.pkl')

    if os.path.exists( filename ) and not args.overwrite:
        logger.info(f"Found {filename}. Do nothing")
        sys.exit(0)
    elif os.path.exists( filename ) and args.overwrite:
        logger.warning(f"Will overwrite {filename}")    

    calib = MultiClassCalibration( os.path.join( os.path.abspath('../../Workflow/configs'), args.config) +".yaml", selection=args.selection, small=args.small)

    calib.train()
    
    calib.save(filename)

    # where to store plots
    plot_directory = os.path.join(user.plot_directory, "Calibration", *subdirs,  args.config, args.selection)
    os.makedirs(plot_directory, exist_ok=True)
    
    calib.plot_calibration(os.path.join( plot_directory, 
                                         f'calibrator_validation_calibration_multi.png'))
    calib.plot_IsotonicRegression(os.path.join( plot_directory, 
                                                f'calibrator_validation_IsoReg_multi.png'))
