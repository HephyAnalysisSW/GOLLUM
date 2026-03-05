import os
import numpy as np
import matplotlib.pyplot as plt
import argparse as ap
import common.user as user
import common.syncer as syncer

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

parser = ap.ArgumentParser(description="Generate loss curve plots from training logs.")
parser.add_argument("-r","--root-dir", required=True, help="Directory containing model subdirectories with loss_curve.txt files")
parser.add_argument("-v","--version-prefix", required=True, help="Prefix for plot titles")

args = parser.parse_args()

root_dir = args.root_dir
version_prefix = args.version_prefix
out_dir = os.path.join(user.plot_directory,f"PNN_losses_{version_prefix}")

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
    title = f"{version_prefix}_{sub}"
    out_pdf = os.path.join(out_dir, f"{title}.pdf")

    # plot
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_loss, linewidth=2.2, label="Train Loss")
    plt.plot(epochs, valid_loss, linewidth=2.2, label="Valid Loss")

    plt.title(title, fontsize=16)
    plt.xlabel("Epoch", fontsize=16)
    plt.ylabel("Loss", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=15)
    plt.grid(False)
    plt.tight_layout()

    plt.savefig(out_pdf)
    plt.savefig(out_pdf.replace(".pdf",".png"))
    plt.close()

    print(f"[ok] {sub}  ->  {out_pdf}/.png")
    n_done += 1

print(f"\nDone. Plots written: {n_done}, skipped: {n_skipped}, output dir: {out_dir}")
