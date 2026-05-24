import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import mplhep as hep
import argparse as ap
import common.user as user
import common.syncer as syncer
from common.helpers import copyIndexPHP


MAKE_PUBLIC_PLOTS = False

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

parser = ap.ArgumentParser(description="Generate TFMC loss curve plots from training logs.")
parser.add_argument("-i","--input", required=True, help="Path to TFMC loss_curve.txt file")
parser.add_argument("-n","--epochs", type=int, help="Number of epochs to plot (from 0). Default: all.")

args = parser.parse_args()

out_dir = os.path.join(user.plot_directory,"TFMC_losses")
os.makedirs(out_dir, exist_ok=True)

epochs, train_loss, valid_loss = read_loss_txt(args.input)

original_epochs = len(epochs)
if args.epochs and args.epochs < original_epochs:
    epochs = epochs[:args.epochs]
    train_loss = train_loss[:args.epochs]
    valid_loss = valid_loss[:args.epochs]

era = None
for lumi_era in ["2016APV","2016", "2017", "2018"]:
    if lumi_era in args.input:
        era = lumi_era
        break

model_name = os.path.dirname(args.input).split("/")[-1]

# title & output name
out_pdf = os.path.join(out_dir, f"TFMC_loss_{model_name}.pdf")

if len(epochs) < original_epochs:
    out_pdf = out_pdf.replace(".pdf",f"_first{len(epochs)}.pdf")

# plot
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(epochs, train_loss, linewidth=2.2, label="Train Loss (rescaled)", color="#3f90da")
ax.plot(epochs, valid_loss, linewidth=2.2, label="Validation Loss", color="#ffa90e")

# ax.set_title(title, fontsize=16)
ax.set_xlabel("Epoch", fontsize=16)
ax.set_ylabel("Loss", fontsize=16)
ax.tick_params(labelsize=14)

legend_title = "TFMC"
if len(epochs) < original_epochs:
    legend_title = f"TFMC (first {len(epochs)}/{original_epochs})"

ax.legend(fontsize=15, frameon=False, title=legend_title, title_fontsize=13, loc="best")
ax.grid(False)
hep.cms.label("Preliminary" if MAKE_PUBLIC_PLOTS else "Internal", data=False, ax=ax, loc=0, fontsize=14, year=era)

plt.tight_layout()
plt.savefig(out_pdf, bbox_inches='tight')
plt.savefig(out_pdf.replace(".pdf",".png"), dpi=240, bbox_inches='tight')
plt.close(fig)

print(f"[ok] {out_pdf} + *.png")

copyIndexPHP(out_dir)
