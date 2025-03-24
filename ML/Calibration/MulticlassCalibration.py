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
import common.datasets_hephy as datasets_hephy

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

    def train( self, probability_training = False):
        logger.info(f"Training: Load data for {self.selection}")

        self.loader = datasets_hephy.get_data_loader(selection=self.selection, n_split=self.n_split)

        # Initialize lists for accumulation
        all_prob = []
        all_weights = []
        all_labels = []

        for batch in tqdm(self.loader, desc="Processing batches"):
            data, weights, labels = self.loader.split(batch)  # Unpack batch

            prob = self.classifier.predict(data, ic_scaling=not probability_training)  # Predict with ic_scaling
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

        if probability_training:
            for label in range(4):
                mask = (all_labels == label)
                total_weight = all_weights[mask].sum()
                if total_weight > 0:
                    all_weights[mask] /= total_weight

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

    @classmethod
    def load( cls, file_name ):
        new_instance = cls()
        with open(file_name, 'rb') as file:
            new_instance.iso_reg = pickle.load(file)
        return new_instance

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
            prob_true_uncalib, prob_pred_uncalib, _, _= weighted_calibration(truth, 
                                                                        uncalib_pred, 
                                                                        nbins=10, 
                                                                        weights = self.data_for_plot['weight'])
            prob_true_calib, prob_pred_calib, _, _ = weighted_calibration(truth, 
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

    def plot_calibration_root(self, file_name):
        import ROOT
        import os
        import common.helpers
        import common.syncer
        import logging
        logger = logging.getLogger(__name__)
        
        # Import soft_colors, label_encoding, and plot_styles from common.data_structure.
        from common.data_structure import label_encoding, plot_styles

        soft_colors = [
            ROOT.TColor.GetColor("#779ECB"),  # Soft blue
            ROOT.TColor.GetColor("#03C03C"),  # Teal green
            ROOT.TColor.GetColor("#B39EB5"),  # Light purple
            ROOT.TColor.GetColor("#FFB347"),  # Soft orange
            ROOT.TColor.GetColor("#FFD1DC"),  # Pastel pink
            ROOT.TColor.GetColor("#AEC6CF"),  # Muted cyan
            ROOT.TColor.GetColor("#CFCFC4"),  # Light gray
            ROOT.TColor.GetColor("#77DD77")   # Pastel green
        ]

        # Define axis range dictionaries per label.
        # Each label key (0,1,2,3) has separate 'main' and 'inset' ranges.
        ranges_per_label = {
            0: {
                'main':  {'xmin': 0., 'xmax': .15, 'ymin': 0., 'ymax': .15},
                'inset': {'xmin': 0,     'xmax': 0.02,   'ymin': 0,     'ymax': 0.02}
            },
            1: {
                'main':  {'xmin': 0., 'xmax': 1., 'ymin': 0., 'ymax': 1.},
                'inset': {'xmin': 0,     'xmax': 0.25, 'ymin': 0,     'ymax': 0.25}
            },
            2: {
                'main':  {'xmin': 0., 'xmax': 1., 'ymin': 0., 'ymax': 1.},
                'inset': {'xmin': 0,     'xmax': 0.2,   'ymin': 0,     'ymax': 0.2}
            },
            3: {
                'main':  {'xmin': 0., 'xmax': .1, 'ymin': 0., 'ymax': .1},
                'inset': {'xmin': 0,     'xmax': 0.2,   'ymin': 0,     'ymax': 0.2}
            }
        }
        
        # Load ROOT TDR style.
        dir_path = os.path.dirname(os.path.realpath(__file__))
        ROOT.gROOT.LoadMacro(os.path.join(dir_path, "../../common/scripts/tdrstyle.C"))
        ROOT.setTDRStyle()
        
        logger.info("Started plotting calibration with ROOT (one figure per class)")
        
        # Loop over the four classes.
        for idx in range(4):
            # Get the configured ranges for this label.
            this_range = ranges_per_label[idx]
            
            # Create a new canvas for each class.
            c = ROOT.TCanvas("c_%d" % idx, "Calibration Plot for Class %d" % idx, 800, 600)
            
            # Create a frame for the main pad using the main ranges.
            frame = c.DrawFrame(this_range['main']['xmin'], this_range['main']['ymin'],
                                this_range['main']['xmax'], this_range['main']['ymax'])
            # Set the x-axis title to the TeX label from plot_styles.
            # For example, if idx==0 then label_encoding[0] gives "htautau" and
            # plot_styles["htautau"]['tex'] gives "H#rightarrow#tau#tau".
            class_str = label_encoding[idx]
            tex_label = plot_styles[class_str]['tex']
            frame.GetXaxis().SetTitle(tex_label)
            frame.GetYaxis().SetTitle("true DCR")
            frame.Draw("axis same")
            
            # Get data for class idx.
            uncalib_pred = self.data_for_plot['prob'][:, idx]
            truth = (self.data_for_plot['label'] == idx).astype(float)
            calib_pred = self.predict(self.data_for_plot['prob'])[:, idx]
            
            # --- Replace the weighted_calibration calls:
            prob_true_uncalib, prob_pred_uncalib, err_true_uncalib, err_pred_uncalib = weighted_calibration(
                truth, uncalib_pred, nbins=20, weights=self.data_for_plot['weight'], x_min = this_range['main']['xmin'], x_max = this_range['main']['xmax'])
            prob_true_calib, prob_pred_calib, err_true_calib, err_pred_calib = weighted_calibration(
                truth, calib_pred, nbins=20, weights=self.data_for_plot['weight'], x_min = this_range['main']['xmin'], x_max = this_range['main']['xmax'])

            # --- Replace the TGraph creation for uncalibrated curve:
            mask_uncalib = (prob_pred_uncalib == 0.) & (prob_true_uncalib == 0.)
            x_uncalib = prob_pred_uncalib[~mask_uncalib]
            y_uncalib = prob_true_uncalib[~mask_uncalib]
            ex_uncalib = err_pred_uncalib[~mask_uncalib]
            ey_uncalib = err_true_uncalib[~mask_uncalib]
            g_uncalib = ROOT.TGraphErrors(len(x_uncalib))
            for i, (x, y, ex, ey) in enumerate(zip(x_uncalib, y_uncalib, ex_uncalib, ey_uncalib)):
                g_uncalib.SetPoint(i, x, y)
                g_uncalib.SetPointError(i, ex, ey)

            # --- And similarly, for the calibrated curve:
            mask_calib = (prob_pred_calib == 0.) & (prob_true_calib == 0.)
            x_calib = prob_pred_calib[~mask_calib]
            y_calib = prob_true_calib[~mask_calib]
            ex_calib = err_pred_calib[~mask_calib]
            ey_calib = err_true_calib[~mask_calib]
            g_calib = ROOT.TGraphErrors(len(x_calib))
            for i, (x, y, ex, ey) in enumerate(zip(x_calib, y_calib, ex_calib, ey_calib)):
                g_calib.SetPoint(i, x, y)
                g_calib.SetPointError(i, ex, ey)


            #
            ## Remove bins where both predicted and true values are zero.
            #mask_uncalib = (prob_pred_uncalib == 0.) & (prob_true_uncalib == 0.)
            #mask_calib   = (prob_pred_calib == 0.) & (prob_true_calib == 0.)
            #x_uncalib = prob_pred_uncalib[~mask_uncalib]
            #y_uncalib = prob_true_uncalib[~mask_uncalib]
            #x_calib   = prob_pred_calib[~mask_calib]
            #y_calib   = prob_true_calib[~mask_calib]
            #
            ## Create TGraphs for the calibration curves.
            #g_uncalib = ROOT.TGraph(len(x_uncalib))
            #for i, (x, y) in enumerate(zip(x_uncalib, y_uncalib)):
            #    g_uncalib.SetPoint(i, x, y)
            #g_calib = ROOT.TGraph(len(x_calib))
            #for i, (x, y) in enumerate(zip(x_calib, y_calib)):
            #    g_calib.SetPoint(i, x, y)
            
            # Style the graphs using soft_colors.
            g_uncalib.SetMarkerStyle(20)
            g_uncalib.SetMarkerColor(soft_colors[0])
            g_uncalib.SetLineColor(soft_colors[0])
            g_uncalib.SetLineWidth(2)
            g_calib.SetMarkerStyle(20)
            g_calib.SetMarkerColor(soft_colors[1])
            g_calib.SetLineColor(soft_colors[1])
            g_calib.SetLineWidth(2)
            
            # Draw the graphs on the main pad.
            g_uncalib.Draw("LP SAME")
            g_calib.Draw("LP SAME")
            
            # Draw the identity line y=x.
            line = ROOT.TLine(this_range['main']['xmin'], this_range['main']['xmin'],
                               this_range['main']['xmax'], this_range['main']['xmax'])
            line.SetLineStyle(2)
            line.Draw("SAME")
            
            # Add a legend.
            leg = ROOT.TLegend(0.2, 0.75, 0.55, 0.85)
            leg.SetBorderSize(0)
            leg.SetFillStyle(0)
            leg.AddEntry(g_uncalib, "before calibration", "lp")
            leg.AddEntry(g_calib, "after calibration", "lp")
            leg.Draw("SAME")
            
            # (No additional title is added; the x-axis title already carries the TeX label.)
            
            # Create an inset pad with a zoomed view.
            inset = ROOT.TPad("inset_%d" % idx, "", 0.625, 0.175, 0.925, 0.475)
            inset.SetFillStyle(0)  # transparent background
            inset.Draw()
            inset.cd()
            
            # Draw a frame for the inset using the inset ranges.
            inset_frame = ROOT.gPad.DrawFrame(this_range['inset']['xmin'], this_range['inset']['ymin'],
                                              this_range['inset']['xmax'], this_range['inset']['ymax'])
            # Set the inset's axis label and title sizes to match the main pad.
            main_label_size = frame.GetXaxis().GetLabelSize()
            main_title_size = frame.GetXaxis().GetTitleSize()
            inset_frame.GetXaxis().SetLabelSize(main_label_size)
            inset_frame.GetYaxis().SetLabelSize(main_label_size)
            inset_frame.GetXaxis().SetTitleSize(main_title_size)
            inset_frame.GetYaxis().SetTitleSize(main_title_size)
            inset_frame.Draw("axis same")
            
            # Draw the same calibration curves and identity line in the inset.
            g_uncalib.Draw("LP SAME")
            g_calib.Draw("LP SAME")
            line_inset = ROOT.TLine(this_range['inset']['xmin'], this_range['inset']['xmin'],
                                    this_range['inset']['xmax'], this_range['inset']['xmax'])
            line_inset.SetLineStyle(2)
            line_inset.Draw("SAME")

            frame.Draw("axis same")
            
            # Return to the main canvas.
            c.cd()
            c.Update()
            
            # Save the canvas to a file; append _classX to the filename.
            base, _ = os.path.splitext(file_name)
            out_file = "%s_class%d%s" % (base, idx, ".png")
            c.SaveAs(out_file)
            logger.info("Saved calibration plot for class %d to %s", idx, out_file)
            out_file = "%s_class%d%s" % (base, idx, ".pdf")
            c.SaveAs(out_file)
            logger.info("Saved calibration plot for class %d to %s", idx, out_file)
        
        common.helpers.copyIndexPHP(os.path.dirname(file_name))
        common.syncer.sync()


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


#def weighted_calibration(true_label, pred_output, nbins=10, weights=None):
#    # reproduces to a large extend 
#    # from sklearn.calibration import calibration_curve
#    # but adds the functionality to have weighted events
#    bins = np.linspace(pred_output.min(), pred_output.max(), nbins+1)
#    if weights is None:
#        weights = np.ones_like(true_label)
#    mask = (pred_output.reshape(-1,1) >= bins[:-1]) & (bins[1:] >= pred_output.reshape(-1,1))
#    true_fraction = []
#    pred_fraction = []
#    for bin_nr in range(nbins):
#        if len(weights[mask[:,bin_nr]]) == 0:
#            true_fraction.append(0.)
#        else:
#            true_fraction.append((weights[mask[:,bin_nr]]*true_label[mask[:,bin_nr]]).sum() /(weights[mask[:,bin_nr]]).sum()+1e-16)
#        if len(pred_output[mask[:,bin_nr]]) == 0:
#            pred_fraction.append(0.)
#        else:
#            pred_fraction.append((pred_output[mask[:,bin_nr]]).mean())
#    prob_true = np.array(true_fraction)
#    prob_pred = np.array(pred_fraction)
#    return prob_true, prob_pred        

def weighted_calibration(true_label, pred_output, nbins=10, weights=None, x_min=None, x_max=None):
    # This function computes weighted calibration curves along with uncertainties.
    # It returns:
    #   prob_true  : weighted true fraction per bin
    #   prob_pred  : weighted mean of pred_output per bin
    #   err_true   : uncertainty on the weighted true fraction (binomial uncertainty)
    #   err_pred   : uncertainty on the weighted predicted mean (standard error)
    
    import numpy as np

    x_min = x_min if not None else pred_output.min()
    x_max = x_max if not None else pred_output.max()

    bins = np.linspace(x_min, x_max, nbins+1)
    if weights is None:
        weights = np.ones_like(true_label)
    
    # Create a boolean mask: each column corresponds to a bin.
    mask = (pred_output.reshape(-1,1) >= bins[:-1]) & (bins[1:] >= pred_output.reshape(-1,1))
    
    true_fraction = []
    pred_fraction = []
    true_uncertainty = []
    pred_uncertainty = []
    
    for bin_nr in range(nbins):
        # Select events in this bin.
        bin_mask = mask[:, bin_nr]
        w_bin = weights[bin_mask]
        
        if len(w_bin) == 0:
            true_fraction.append(0.)
            pred_fraction.append(0.)
            true_uncertainty.append(0.)
            pred_uncertainty.append(0.)
        else:
            w_sum = w_bin.sum()
            # Effective number of events in the bin.
            n_eff = (w_sum ** 2) / ((w_bin ** 2).sum())
            
            # Compute the weighted true fraction (using true_label).
            p_true = (w_bin * true_label[bin_mask]).sum() / w_sum + 1e-16
            true_fraction.append(p_true)
            # Uncertainty on the true fraction, binomial style.
            sigma_true = (p_true * (1 - p_true) / n_eff) ** 0.5
            true_uncertainty.append(sigma_true)
            
            # Compute the weighted mean of the predicted output.
            p_pred = (w_bin * pred_output[bin_mask]).sum() / w_sum
            pred_fraction.append(p_pred)
            # Compute the weighted variance.
            variance = (w_bin * (pred_output[bin_mask] - p_pred)**2).sum() / w_sum
            sigma_pred = (variance / n_eff) ** 0.5
            pred_uncertainty.append(sigma_pred)
    
    prob_true = np.array(true_fraction)
    prob_pred = np.array(pred_fraction)
    err_true  = np.array(true_uncertainty)
    err_pred  = np.array(pred_uncertainty)
    
    return prob_true, prob_pred, err_true, err_pred


if __name__=="__main__":
    import argparse
    import common.syncer
    # Argument parser setup
    parser = argparse.ArgumentParser(description="ML inference.")
    parser.add_argument('--logLevel', action='store', nargs='?', choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'TRACE', 'NOTSET'], default='INFO', help="Log level for logging")
    parser.add_argument("--config", default = "config_reference_v2_sr", help="Path to the config file.")
    parser.add_argument("--selection", default="lowMT_VBFJet", help="Which selection?")
    parser.add_argument("--save", action="store_true", help="Save the ML predictions for the simulation.")
    parser.add_argument("--small", action="store_true", help="Run a subset.")
    parser.add_argument("--probability_training", action="store_true", help="Run a subset.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    parser.add_argument("--postfix", default = "", type=str,  help="Append this to the fit result.")

    args = parser.parse_args()

    logger  = get_logger(args.logLevel, logFile = None)

    if args.probability_training:
        args.postfix = "prob" if args.postfix=="" else "prob_"+args.postfix

    subdirs = [args.selection, args.postfix]

    # Where to store the training
    model_directory = os.path.join(user.model_directory, "Calibration", *subdirs,  args.config, args.selection)
    os.makedirs(model_directory, exist_ok=True)
    filename = os.path.join( model_directory, f'calibrator_multi.pkl')

    if os.path.exists( filename ) and not args.overwrite:
        logger.info(f"Found {filename}. No training!")
        calib = MultiClassCalibration.load( filename )
        sys.exit(0)
    elif os.path.exists( filename ) and args.overwrite:
        logger.warning(f"Will overwrite {filename}")    

    calib = MultiClassCalibration( os.path.join( os.path.abspath('../../Workflow/configs'), args.config) +".yaml", selection=args.selection, small=args.small)

    calib.train(probability_training=args.probability_training)
    
    calib.save(filename)

    # where to store plots
    plot_directory = os.path.join(user.plot_directory, "Calibration", *subdirs,  args.config, args.selection)
    os.makedirs(plot_directory, exist_ok=True)
    
    #calib.plot_calibration(os.path.join( plot_directory, 
    #                                     f'calibrator_validation_calibration_multi.png'))
    calib.plot_calibration_root(os.path.join( plot_directory, 
                                         f'multicalibrator_validation_calibration_multi.png'))
    calib.plot_IsotonicRegression(os.path.join( plot_directory, 
                                                f'multicalibrator_validation_IsoReg_multi.png'))

    common.syncer.sync()
