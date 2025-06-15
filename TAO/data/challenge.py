''' Here we interface to Gollum. We reuse IC and the Scalar
'''
import sys
sys.path.insert(0, '..')
import os
import common.datasets_hephy as datasets_hephy
import common.user as user

def load_training_data( selection="lowMT_VBFJet", small = False, use_ic = True, use_scaler = True, rng=None):

    n_split = 10 if not small else 10000

    result = {'loader': datasets_hephy.get_data_loader( selection=selection, selection_function=None, n_split=n_split)}

    if use_ic:
        from ML.IC.IC import InclusiveCrosssection
        ic = InclusiveCrosssection.load(os.path.join(user.model_directory, "IC", "IC_"+selection+'.pkl'))
        print("We use this IC:")
        print(ic)
        result['weight_sums'] = ic.weight_sums

    # Do we use a Scaler?
    if use_scaler:
        from ML.Scaler.Scaler import Scaler
        scaler = Scaler.load(os.path.join(user.model_directory, "Scaler", "Scaler_"+selection+'.pkl'))
        result["X_mean"] = scaler.feature_means
        result["X_std"]  = scaler.feature_variances

        print("We use this scaler:")
        print(scaler)

    return result 
