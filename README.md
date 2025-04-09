# Higgs ML Uncertainty Challenge -- HEPHY

## Hardware

CPU model: Intel(R) Xeon(R) Gold 6138 CPU @ 2.00GHz
Architecture: x86\_64
number of CPU cores: 1
memory: 20GB
GPU: not needed

Minimum requirements:
number of CPU cores: 1
memory: 20GB
GPU: not needed

## OS

CentOS Linux 7

## 3rd-party software and environment setup

TensorFlow: `pip install tensorflow`
imunuit: `pip install iminuit`

To ensure a consistent environment, we recommend using **Conda**. You can create the required environment using:

```bash
conda env create -f environment.yml
conda activate uncertainty_challenge_new
```

# Dataset Preparation

This section describes how to generate the train and test datasets used in this project. All scripts related to dataset generation are in the `Dataset_Preparation/` folder.

## 1. Train Set

To create the train set, you will need the following four scripts:

- `Higgs_Datasets_Train.py`
- `Higgs_Datasets_Train_Generation.py`
- `derived_quantities.py`
- `systematics.py`

**Usage**: Simply run:

```bash
python Higgs_Datasets_Train_Generation.py
```

- `input_directory` specifies the path to the dataset (downloaded from the Higgs Uncertainty Challenge).
- Six parameters (`tes`, `jes`, `soft_met`, `ttbar_scale`, `diboson_scale`, `bkg_scale`) define systematic uncertainties. Modifying these values generates different systematic variants of the training dataset.
- `hdf5_filename` is the path and filename for the output `.h5` file.

The resulting dataset is an `.h5` file with shape \((N, 30)\):
- The first 16 columns are **primary features**. The next 12 columns are **derived features** (see [2410.02867] for details).
- The 29th column is the **event weight**.
- The 30th column is the **label**, where:
  - **0** = htautau  
  - **1** = ztautau  
  - **2** = ttbar  
  - **3** = diboson  

---

## 2. Test Set

Generating the test set follows the same structure, using:

- `Higgs_Datasets_Test.py`
- `Higgs_Datasets_Test_Generation.py`
- `derived_quantities.py`
- `systematics.py`

**Usage**: Simply run:

```bash
python Higgs_Datasets_Test_Generation.py
```

All parameters (systematic values, directories, etc.) are set in the same way as for the train set.

## 3. Applying Selection Cuts
After generating train or test datasets (including systematic variations), you can apply additional selection cuts to obtain data in specific signal regions. These cuts are defined in common/selections.py and can be combined to form more detailed filtering criteria.

Use the copy_with_selection.py script to filter the .h5 dataset. For example:
```bash
python copy_with_selection.py \
  --files /path/to/dataset.h5 \
  --target-dir /path/to/output/ \
  --selection lowMT_VBFJet \
  --n-batches 100 \
  --overwrite
```
--files: One or more input HDF5 files (wildcards like *.h5 are allowed).

--target-dir: Where filtered files will be saved.

--selection: The name of the selection, defined in common/selections.py (e.g., lowMT_VBFJet).

--n-batches: Number of batches to read the input file in (helps reduce memory usage).

--overwrite: Overwrite existing files in the target directory.

By changing --selection, you can apply different cuts (e.g., highMT, noVBFJet, ptH0to100) either alone or in combination.

This process yields new .h5 files, each containing only events that pass the specified selection criteria.


## ML models

There are two models: 
- TensorFlow multiclassifier (TFMC)
- Parametric neural network (PNN)

### Training

The models are pre-trained, so no training needed.


### Inference

The TFMC and PNN are used together to infer the interval of the signal strength. The evaluation is performed in the `model.py`, with the input of a config file `configs/config_submission.yaml`.

#### Config files

The config file `configs/config_submission.yaml` is hardcoded in `model.py`. The important sections are the following:

- `Tasks`: This specifies which tasks to run, including multiclassifier and the parametric neural network for all the processes. This section should not be changed.
- `Selections`: The framework applies selections (defined in `common/selections.py`) on events to categorize them into different regions. This specifies which regions to use in the signal strength inference.
- `CSI`: This sets whether to use cubic spline interpolation for inclusive cross section. It should not be changed.
- `MultiClassifier/htautau/ztautau/ttbar/diboson`: The ML architechture and model paths for each regions are set in these sections. Different files are provided for different tasks. For multiclassifier, `model_path` and `calibration` are provided, the `calibration` is optional. For PNN in different processes, the inclusive cross section parametrization file `icp_file` and `model_path` are provided.

#### Trained models

The trained models are stored in `models/*Task*/*selection*/*specifics*/`. The `*Task*`, `*selection*`, and `*specifics*` are explained below:
- `Task`: The `Tasks` in the config file, includes `MultiClassifier`, `htautau`, `ztautau`, `ttbar`, and `diboson`.
- `selection`: The `Selections` in the config file, includes `lowMT_VBFJet`, `lowMT_noVBFJet_ptH100`, and `highMT_VBFJet`.
- `specifics`: The stored files includes the trained model path (`model_path`), calibration files for multiclassifier (`calibration`), and inclusive cross section parametrization file (`icp_file`).

In addition, the `CSI` files for training data is used as well in the prediction. Those files are saved in `data/tmp_data/`.

## Side effects

- Running `predict.py` produces a `results.json` under the `SUBMISSION_DIR` provided in the arguments. The json file will be overwritten if the file already exists.

## Key assumptions

- The framework applies selections on the data before processing it. The selections are defined in `common/selections.py`. A set of selections can be used when infering the signal strength, as specified in `configs/config_submission.yaml`. When running the framework, it assumes non-zero events from all selections.
- The script `predict.py` should be run under the main directory.
