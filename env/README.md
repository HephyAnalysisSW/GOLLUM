### setting up the environment
```
conda activate /groups/hephy/cms/robert.schoefbeck/conda/envs/hephy-ml-gpu
```

Create it with:
```
mamba create -n hephy-ml-gpu2 -c conda-forge python=3.10   root numpy scipy pandas matplotlib seaborn scikit-learn scikit-optimize xgboost lightgbm iminuit requests tabulate tqdm uproot uproot3
mamba activate hephy-ml-gpu2
mamba install -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=11.8
python -m pip install -U pip
pip install --index-url https://download.pytorch.org/whl/cu118   torch torchvision torchaudio
pip install -U "jax"
python -m pip install -U "tensorflow[and-cuda]"
```
