# BIT calibration

The code in this folder is used to verify if the trained BITs are calibrated, i.e. if they are properly learning the each of the derivative terms.

It compares directly the average (truth - predicted) in bins of the predicted value, instead of its projections in bins of a single kinematic variable. 

This integrates over the possible latent space for events in that bin and removes effects from the noisy mapping from latent to reconstruction-level space. This is similar to the integration done by the BIT training.

A well-calibrated BIT will have a line at 0.

The steps to obtain the plots and use their output to do post-training calibration of the BIT are described below (run the different scripts with `-h` to see the different CLI options).

1. extract arrays with truth and predicted values with `calibration_runner.py`.

2. plot the values with `calibration_plots.py`.

`calibration_plots.py` also allows deriving binned calibration factors to rescale the output of each node such that the truth and predict binned distributions match. These can then be applied using the functionality in `binned_calibration.py`. NB: this output is **heavily dependent on the binning of the calibration plots**.

