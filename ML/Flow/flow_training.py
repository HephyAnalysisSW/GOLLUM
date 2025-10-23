#!/usr/bin/env python
import sys, os, glob, re
sys.path.insert(0, '..')
sys.path.insert(0, '../..')
import importlib
from tqdm import tqdm
import torch

from Flow import Flow
import common.user as user
import common.helpers as helpers
import common.datasets_hephy as datasets_hephy
import common.data_structure as data_structure
import common.syncer

import argparse
argParser = argparse.ArgumentParser()
argParser.add_argument('--overwrite', action='store_true')
argParser.add_argument("--selection", default="lowMT_VBFJet")
argParser.add_argument("--n_split", type=int, default=10)
argParser.add_argument("--every", type=int, default=5)
argParser.add_argument("--training", default="v2")
argParser.add_argument("--config", default="flow_v1")
argParser.add_argument("--configDir", default="configs")
argParser.add_argument('--small', action='store_true')
args = argParser.parse_args()

# load config
config = importlib.import_module(f"{args.configDir}.{args.config}")

# optional scaler
if getattr(config, 'use_scaler', False):
    from ML.Scaler.Scaler import Scaler
    scaler = Scaler.load(os.path.join(
        user.model_directory, f"Scaler/Scaler_{args.selection}.pkl"))
    config.feature_means     = scaler.feature_means
    config.feature_variances = scaler.feature_variances

# directories
model_dir = os.path.join(user.model_directory,
                         "Flow", args.selection, args.config,
                         args.training + ("_small" if args.small else ""))
os.makedirs(model_dir, exist_ok=True)

plot_dir  = os.path.join(user.plot_directory,
                         "Flow", args.selection, args.config,
                         args.training + ("_small" if args.small else ""))
helpers.copyIndexPHP(plot_dir)

# --- instantiate or load ---
if not args.overwrite:
    try:
        print(f"Loading checkpoint from {model_dir}")
        flow = Flow.load(model_dir, config=config)
    except FileNotFoundError:
        print("No checkpoint found → new model")
        flow = Flow(config=config, model_dir=model_dir)
else:
    flow = Flow(config=config, model_dir=model_dir)

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# note: if Flow wraps a nn.Module, you may need
# flow.model.to(device); flow.embed_net.to(device)
flow.device = device

# optimizer + scheduler
params = list(flow.model.parameters()) + list(flow.embed_net.parameters())
optimizer = torch.optim.Adam(params,
                             lr=config.learning_rate,
                             weight_decay=getattr(config, 'weight_decay', 0.0))
flow.optimizer = optimizer

if hasattr(config, 'lr_scheduler'):
    sched_cfg = config.lr_scheduler.copy()
    sched_type = sched_cfg.pop('type')
    Scheduler = getattr(torch.optim.lr_scheduler, sched_type)
    scheduler = Scheduler(optimizer, **sched_cfg)
    flow.scheduler = scheduler

# data loader
flow.load_training_data(datasets_hephy,
                        args.selection,
                        n_split=(args.n_split if not args.small else 10))

max_batch = 1 if args.small else -1

# determine starting epoch
if not args.overwrite:
    ckpt_epochs = []
    for fn in glob.glob(os.path.join(model_dir, "flow_epoch_*.pt")):
        m = re.search(r"flow_epoch_(\d+)\.pt$", fn)
        if m: ckpt_epochs.append(int(m.group(1)))
    if os.path.exists(os.path.join(model_dir, "flow_final.pt")):
        starting_epoch = max(ckpt_epochs) + 1 if ckpt_epochs else 0
    else:
        starting_epoch = max(ckpt_epochs) + 1 if ckpt_epochs else 0
else:
    starting_epoch = 0

# -- training loop --
for epoch in range(starting_epoch, config.n_epochs):
    # print LR
    lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch}/{config.n_epochs}, lr={lr:.3e}")

    # train (and optionally accumulate histograms)
    do_hist = (epoch % args.every == 0)
    true_hist, pred_hist = flow.train_one_epoch(
        max_batch=max_batch,
        accumulate_histograms=do_hist)

    # step scheduler
    if hasattr(flow, 'scheduler'):
        flow.scheduler.step()

    # save checkpoint
    flow.save(epoch)

    # plot if requested
    if do_hist:
        flow.plot_convergence_root(
            true_hist, pred_hist,
            epoch, plot_dir,
            data_structure.feature_names)
        common.syncer.makeRemoteGif(plot_dir,
                                    pattern="epoch_*.png",
                                    name="epoch")
        common.syncer.makeRemoteGif(plot_dir,
                                    pattern="norm_epoch_*.png",
                                    name="norm_epoch")

    # sync to remote
    common.syncer.sync()

# final sync
common.syncer.sync()

