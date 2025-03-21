# README for Data Loader of Higgs Dataset with Systematic Variations

### Overview

I have created nine HDF5 files located at `/eos/vbc/group/mlearning/data/Higgs_uncertainty/input_data/train_dataset`. These files have been processed through systematic variations, post-processing, and the calculation of 12 derived quantities, before being saved as `.h5` files. Each HDF5 file contains the following information:

- **Data**: Each event includes 16 primary features and 12 derived features.
- **Labels**: Labels are either `1` (signal) or `0` (background).
- **Detailed Labels**: Detailed classification of events including categories `ttbar`, `ztautau`, `htautau`, and `diboson`.
- **Event Weights**: Event weights, with three Weight Nuisance Parameters each fixed to + 1 sigma.

### Systematic Factors for Each HDF5 File

1. **`syst_train_set_test0.h5`** (with no variation):
   - TES = 1.0
   - JES = 1.0
   - Soft MET = 0
   - ttbar Scale = 1.0
   - Diboson Scale = 1.0
   - Background Scale = 1.0

2. **`syst_train_set_test1.h5`** (with +1 sigma for all 6 Nuisance Parameters):
   - TES = 1.01
   - JES = 1.01
   - Soft MET = 3
   - ttbar Scale = 1.25
   - Diboson Scale = 1.025
   - Background Scale = 1.01

3. **`syst_train_set_test2.h5`** (with TES -1 sigma):
   - TES = 0.99
   - JES = 1.01
   - Soft MET = 3
   - ttbar Scale = 1.25
   - Diboson Scale = 1.025
   - Background Scale = 1.01

4. **`syst_train_set_test3.h5`** (with JES -1 sigma):
   - TES = 1.01
   - JES = 0.99
   - Soft MET = 3
   - ttbar Scale = 1.25
   - Diboson Scale = 1.025
   - Background Scale = 1.01

5. **`syst_train_set_test4.h5`** (TES and JES -1 sigma):
   - TES = 0.99
   - JES = 0.99
   - Soft MET = 3
   - ttbar Scale = 1.25
   - Diboson Scale = 1.025
   - Background Scale = 1.01

6. **`syst_train_set_test5.h5`** (Soft MET at +0.5 sigma, TES and JES +1 sigma):
   - TES = 1.01
   - JES = 1.01
   - Soft MET = 1.5
   - ttbar Scale = 1.25
   - Diboson Scale = 1.025
   - Background Scale = 1.01

7. **`syst_train_set_test6.h5`** (TES -1 sigma, JES +1 sigma, Soft MET +0.5 sigma):
   - TES = 0.99
   - JES = 1.01
   - Soft MET = 1.5
   - ttbar Scale = 1.25
   - Diboson Scale = 1.025
   - Background Scale = 1.01

8. **`syst_train_set_test7.h5`** (TES +1 sigma, JES -1 sigma, Soft MET +0.5 sigma):
   - TES = 1.01
   - JES = 0.99
   - Soft MET = 1.5
   - ttbar Scale = 1.25
   - Diboson Scale = 1.025
   - Background Scale = 1.01

9. **`syst_train_set_test8.h5`** (TES and JES -1 sigma, Soft MET +0.5 sigma):
   - TES = 0.99
   - JES = 0.99
   - Soft MET = 1.5
   - ttbar Scale = 1.25
   - Diboson Scale = 1.025
   - Background Scale = 1.01

### Data Loader and Visualization

The corresponding data loader scripts are **`data_loader.py`** and **`test_data_loader.py`**. These scripts include functionality for basic dataset loading, sample management, and batch handling. Additional features such as preprocessing can be easily modified or added as required.

In addition, I have tested and optimized the visualization code. The scripts for visualization are **`visualization.py`** and **`test_visualization.py`**. These include functionality for:

- Dataset examination and description
- Data histograms
- Feature correlations
- Pair plots
- Stacked histograms
- Correlation of systematic uncertainties
- Histograms including systematic variations 
... etc.

These visualization tools provide a comprehensive overview of both the nominal and systematic datasets.

