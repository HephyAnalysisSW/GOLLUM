# HEPHY's contribution to the Higgs ML Uncertainty Challenge

## Getting started

### Setup
`git clone git@github.com:HephyAnalysisSW/HEPHY-uncertainty`

### Every time (make aliases for these commands)
`conda activate /groups/hephy/mlearning/conda_envs/uncertainty_challenge`
### Test it
`ipython -i BPT/bpt_training.py`

### Interface
To control the training and prediction of ML algos used in the challenge, we aim to use one script ([Interface.py](https://github.com/HephyAnalysisSW/HEPHY-uncertainty/blob/master/Workflow/Interface.py)) to perform the training and prediction for all ML algos. A [config yaml file](https://github.com/HephyAnalysisSW/HEPHY-uncertainty/blob/master/Workflow/config.yaml), which includes all of the configurable parameters for the ML algos, is read by the script.

The yaml file should include all of the ML algorithms we use in the challenge. For a ML classs to be used in the script, the following design should be followed:
- It could be initialized by one single argument (could be a dictionary or a module)
- It should have the `train` and `predict` function, with the data loading process handled
- It should be registered in [Workflow/networks/Models.py](https://github.com/HephyAnalysisSW/HEPHY-uncertainty/blob/master/Workflow/networks/Models.py)

In the yaml file, the first section `Tasks` controls which ML algos to run. Only the algorithm(s) specified in the `Tasks` section will be excuted by the script. All of the remaining sections after the `Tasks` section records the configurable parameters for each ML algorithms:
- `module`: the name of the ML class registered in [Workflow/networks/Models.py](https://github.com/HephyAnalysisSW/HEPHY-uncertainty/blob/master/Workflow/networks/Models.py)
- `model`: the file name of the trained model. The full path will be `common.user.model_directory` + `model` (This probably needs to be changed for prediction so we can run models trained by others)
- `config`: the config file that could be used to initialize the ML algo. Different configs are available in [ML/configs](https://github.com/HephyAnalysisSW/HEPHY-uncertainty/tree/master/ML/configs)
- `selection`: the selections to used for the input data, defined in [common/selections.py](https://github.com/HephyAnalysisSW/HEPHY-uncertainty/blob/master/common/selections.py)
- `hyper_param`: hyper parameters for the training of ML algorithms. (Will be implemented after more ML algorithms are included.)

#### Usage
```
cd Workflow
python Interface.py --train # to train
python Interface.py --predict # to predict
```
This is only tested on ICP now. The prediction simply runs `icp.predict((1.,))` and `icp.predict((2.,))`. Then the results are printed out. Will modify this part later when we decide how to take all the information for ML algos and calculate the limit.
