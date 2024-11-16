# HEPHY's contribution to the Higgs ML Uncertainty Challenge

## Getting started
### Setup
`git clone git@github.com:HephyAnalysisSW/HEPHY-uncertainty`
### Every time (make aliases for these commands)
`conda activate /groups/hephy/mlearning/conda_envs/uncertainty_challenge`

### Creating the environment

I created the common env using

`mamba create -p /groups/hephy/mlearning/conda_envs/uncertainty_challenge -c conda-forge -c pyg python=3.10 root=6.28.0`

and a subsequent 

`mamba activate /groups/hephy/mlearning/conda_envs/uncertainty_challenge`

and

`mamba install -c conda-forge --file requirements.txt`

