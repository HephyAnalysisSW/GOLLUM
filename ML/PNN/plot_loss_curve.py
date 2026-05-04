import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import mplhep as hep
import argparse as ap
import common.user as user
import common.syncer as syncer
from common.helpers import copyIndexPHP

def read_loss_txt(path):
    epochs, train_loss, valid_loss = [], [], []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            # epoch lr train_loss valid_loss
            epochs.append(int(parts[0]))
            train_loss.append(float(parts[2]))
            valid_loss.append(float(parts[3]))
    return np.array(epochs), np.array(train_loss), np.array(valid_loss)

matplotlib.use("Agg")
plt.style.use("petroff10")
hep.style.use("CMS")

parser = ap.ArgumentParser(description="Generate loss curve plots from training logs.")
parser.add_argument("-r","--root-dir", required=True, help="Directory containing model subdirectories with loss_curve.txt files")
parser.add_argument("-v","--version-prefix", required=True, help="Prefix for plot titles")

args = parser.parse_args()

root_dir = args.root_dir
version_prefix = args.version_prefix
out_dir = os.path.join(user.plot_directory,"PNN_losses", version_prefix)

os.makedirs(out_dir, exist_ok=True)

subdirs = sorted(
    d for d in os.listdir(root_dir)
    if os.path.isdir(os.path.join(root_dir, d))
)

if not subdirs:
    raise RuntimeError(f"No subfolders found under: {root_dir}")

n_done, n_skipped = 0, 0

for sub in subdirs:
    loss_path = os.path.join(root_dir, sub, "loss_curve.txt")
    if not os.path.isfile(loss_path):
        print(f"[skip] {sub}: loss.txt not found")
        n_skipped += 1
        continue

    epochs, train_loss, valid_loss = read_loss_txt(loss_path)

    # title & output name
    title = f"{sub}"
    out_pdf = os.path.join(out_dir, f"{sub}.pdf")

    # plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(epochs, train_loss, linewidth=2.2, label="Train Loss", color="#3f90da")
    ax.plot(epochs, valid_loss, linewidth=2.2, label="Validation Loss", color="#ffa90e")

    # ax.set_title(title, fontsize=16)
    ax.set_xlabel("Epoch", fontsize=16)
    ax.set_ylabel("Loss", fontsize=16)
    ax.tick_params(labelsize=14)

    import re
    pattern = r'pnn_(.+?)_(2016APV|2016|2017|2018)_(.+)$'

    # Example usage:
    match = re.match(pattern, sub)

    if match:
        sample_name = match.group(1)          # "TTLep_pow"
        year = match.group(2)                 # "2018"
        nuisance_param = match.group(3)       # "CMS_scale_j_FlavorPureCharm"
        # print(f"Found a match for {sub=}, {sample_name=}, {year=}, {nuisance_param=} ")
    else:
        print(f"Did not find a match for {sub=}")

    from data.plot_options import get_sample_legend, get_nice_parameter_name
    
    sample_legend = "$"+get_sample_legend(sample_name)+"$"
    sample_legend = sample_legend.replace("#","\\")
    param_name = get_nice_parameter_name(nuisance_param)

    ax.legend(fontsize=15, frameon=False, title=sample_legend+f", {param_name}", title_fontsize=13, loc="best")
    ax.grid(False)
    hep.cms.label("Preliminary", data=False, ax=ax, loc=0, fontsize=14, year=year)

    plt.tight_layout()
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.savefig(out_pdf.replace(".pdf",".png"), dpi=1000, bbox_inches='tight')
    plt.close(fig)

    print(f"[ok] {sub}  ->  {out_pdf} + *.png")
    n_done += 1

copyIndexPHP(out_dir)
print(f"\nDone. Plots written: {n_done}, skipped: {n_skipped}, output dir: {out_dir}")
