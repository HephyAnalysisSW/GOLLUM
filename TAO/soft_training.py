#!/usr/bin/env python
import sys, os
sys.path.insert(0, '..')

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import importlib
import argparse

from TAO import SoftForest as Forest, SoftTree as Tree
import common.user as user
import common.syncer
import common.helpers as helpers
from common.logger import get_logger

# -----------------------------------------------------------------------------
# Parser
# -----------------------------------------------------------------------------
argParser = argparse.ArgumentParser(description="Train a soft decision‐tree forest with PyTorch")
argParser.add_argument('--overwrite',     action='store_true', help="Restart training from scratch")
argParser.add_argument('--every',         type=int, default=1,    help="Update plot every N epochs")
argParser.add_argument('--data',          choices=['toy','challenge'], default='challenge', help="Dataset to use")
argParser.add_argument('--forest_config', type=str, default="configs/tree_softtree_1node.yaml", help="Forest YAML config")
argParser.add_argument('--train_config',  type=str, default="configs/training_softtao.py", help="Training YAML config")
argParser.add_argument('--training',      type=str, default="",     help="Optional postfix for run")
#argParser.add_argument('--quantization',  nargs='*', type=int, choices=[-1,2,3], default=[], help="Bits per depth (empty→no quant.)")
argParser.add_argument('--small',         action='store_true', help="Debug mode: single batch")
argParser.add_argument('--logLevel',      nargs='?', choices=['CRITICAL','ERROR','WARNING','INFO','DEBUG','TRACE','NOTSET'], default='INFO', help="Log level")
args = argParser.parse_args()

logger = get_logger(args.logLevel, logFile=None)

# -----------------------------------------------------------------------------
# Load forest config
# -----------------------------------------------------------------------------
with open(args.forest_config, 'r') as f:
    forest_config = yaml.safe_load(f)
# allow YAML to specify rng_seed; build Generator
forest_config['rng'] = np.random.default_rng(forest_config.get('rng_seed', None))

## process quantization flags
#for i, q in enumerate(args.quantization):
#    args.quantization[i] = None if q < 0 else q
#if len(args.quantization)==1:
#    args.quantization = [ args.quantization[0] ]*(forest_config['max_depth']+1)
#elif len(args.quantization) not in (0, forest_config['max_depth']+1):
#    raise RuntimeError(f"Bad quantization spec: {args.quantization}")
#forest_config['quantization'] = args.quantization

# optional postfix for model/plot dirs
postfix = []
if args.training:
    postfix.append(args.training)
#if args.quantization:
#    qstr = "_".join("None" if q is None else str(q) for q in args.quantization)
#    postfix.append(f"quant_{qstr}")
postfix = "_" + "_".join(postfix) if postfix else ""

# -----------------------------------------------------------------------------
# Load data module
# -----------------------------------------------------------------------------
training_module = importlib.import_module(f"data.{args.data}")
training_data  = training_module.load_training_data(small=args.small, rng=forest_config['rng'])

# -----------------------------------------------------------------------------
# Prepare directories
# -----------------------------------------------------------------------------
model_directory = os.path.join(
    user.model_directory, "softTAO", args.data,
    os.path.splitext(os.path.basename(args.forest_config))[0] + postfix,
    os.path.splitext(os.path.basename(args.train_config))[0] + ("_small" if args.small else "")
)
os.makedirs(model_directory, exist_ok=True)

plot_directory = os.path.join(
    user.plot_directory, "softTAO", args.data,
    os.path.splitext(os.path.basename(args.forest_config))[0] + postfix,
    os.path.splitext(os.path.basename(args.train_config))[0] + ("_small" if args.small else "")
)
helpers.copyIndexPHP(os.path.join(plot_directory, "1D"))

# -----------------------------------------------------------------------------
# Instantiate or load forest
# -----------------------------------------------------------------------------
if args.overwrite:
    start_epoch = 0
    logger.info("→ Overwrite: starting from scratch")
    forest = Forest(
        [ Tree(forest_config) for _ in range(forest_config.get("ntrees",1)) ],
        config=forest_config
    )
else:
    try:
        forest = Forest.load(model_directory)
        # find latest epoch from filenames
        epochs = [
            int(fn.split('_')[-1].split('.')[0])
            for fn in os.listdir(model_directory)
            if fn.startswith("forest_epoch_") and fn.endswith(".pkl")
        ]
        start_epoch = max(epochs) + 1
        logger.info(f"→ Resuming from epoch {start_epoch}")
    except FileNotFoundError:
        start_epoch = 0
        logger.info("→ No checkpoint found, starting at epoch 0")
        forest = Forest(
            [ Tree(forest_config) for _ in range(forest_config.get("ntrees",1)) ],
            config=forest_config
        )

# -----------------------------------------------------------------------------
# Load training config
# -----------------------------------------------------------------------------
with open(args.train_config, 'r') as f:
    train_config = yaml.safe_load(f)

# -----------------------------------------------------------------------------
# Plotting helper
# -----------------------------------------------------------------------------
def plot1D(filename, X, y, y_pred, bins=50, weight=None, title="1D response", text=""):
    import ROOT
    tex = ROOT.TLatex(); tex.SetNDC(); tex.SetTextSize(0.07); tex.SetTextAlign(11)
    if weight is None:
        weight = np.ones_like(y)
    d = X.shape[1]
    cols = min(3,d); rows = (d+cols-1)//cols
    c = ROOT.TCanvas("c1d", title, 300*cols, 300*rows)
    c.Divide(cols, rows)
    for i in range(d):
        h2   = ROOT.TH2F(f"h2_{i}", "", bins, X[:,i].min(), X[:,i].max(), bins, y.min(), y.max())
        hprof= ROOT.TProfile(f"hp_{i}", "", bins, X[:,i].min(), X[:,i].max())
        htrue= ROOT.TProfile(f"ht_{i}", "", bins, X[:,i].min(), X[:,i].max())
        for xi, yi, ypi, wi in zip(X[:,i], y, y_pred, weight):
            h2.Fill(xi, ypi, wi)
            hprof.Fill(xi, ypi, wi)
            htrue.Fill(xi, yi, wi)
        c.cd(i+1)
        h2.SetStats(False); h2.Draw("COLZ")
        htrue.SetLineColor(ROOT.kBlack); htrue.SetLineWidth(2); htrue.Draw("SAME")
        hprof.SetLineColor(ROOT.kRed); hprof.SetMarkerColor(ROOT.kRed); hprof.SetLineWidth(2); hprof.Draw("SAME")
    if text:
        obj = tex.DrawLatex(0.3,0.95,text); obj.Draw()
    c.Update(); c.Print(filename)

# -----------------------------------------------------------------------------
# PyTorch training loop
# -----------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
forest = forest.to(device)
optimizer = optim.Adam(forest.parameters(), lr=train_config['lr'])

if forest_config['mode'] == 'regression':
    criterion = nn.MSELoss(reduction='none')
else:
    criterion = nn.CrossEntropyLoss(reduction='none')

for epoch in range(start_epoch, train_config['n_epochs']):
    logger.info(f"=== Epoch {epoch} ===")
    forest.train()
    running_loss = 0.0

    for i_batch, batch in enumerate(training_data['loader']):
        data, weights, raw_labels = training_data['loader'].split(batch)
        inputs = torch.from_numpy(data).float().to(device)
        sample_w = torch.from_numpy(weights).float().to(device)

        if forest_config['mode'] == 'regression':
            targets = torch.from_numpy(raw_labels).float().to(device)
        else:
            targets = torch.from_numpy(raw_labels).long().to(device)

        optimizer.zero_grad()
        outputs = forest(inputs)
        loss = criterion(outputs, targets)
        loss = (loss * sample_w).mean()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if args.small:
            break

    avg_loss = running_loss / (i_batch + 1)
    logger.info(f"Epoch {epoch} loss: {avg_loss:.4f}")

    # evaluate & plot
    forest.eval()
    with torch.no_grad():
        X_cpu = torch.from_numpy(data).float().to(device)
        preds = forest(X_cpu).cpu().numpy()

    plot1D(
        os.path.join(plot_directory, "1D", f"epoch_{epoch:04d}.png"),
        data, raw_labels, preds,
        weight=weights, text=f"Epoch {epoch:04d}"
    )

    common.syncer.makeRemoteGif(
        os.path.join(plot_directory, "1D"),
        pattern="epoch_*.png", name="epoch"
    )
    common.syncer.sync()
    forest.save(model_directory, epoch)

