import numpy as np
import common.user as user
import common.syncer as syncer
import re 
import logging
import matplotlib.pyplot as plt
import os, sys
import argparse as ap
import glob

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

sys.path.insert(0, '..')
sys.path.insert(0, '../..')
sys.path.insert(0, '../../..')

# for file with loss history from all terms
def sanitize_header_label(label):
    #print(label.translate("'()"))
    new_label = label.removeprefix("train_loss_").removeprefix("valid_loss_")
    new_label = re.sub(r"['(),]","",new_label)
    new_label = new_label.split(" ")
    if len(new_label) > 1:
        if new_label[0] == new_label[1]:
            new_label = f"{new_label[0]}pow2"
        else:
            new_label = f"{new_label[0]}_x_{new_label[1]}"
    else:
        new_label = new_label[0]

    return new_label
    

def plot_bit_losses_all_terms(plot_dir: str, loss_txt_all_terms: str):

    header = []
    blocks = []
    with open(loss_txt_all_terms, "r") as f:
        blocks = [line.strip().split("\t") for line in f.readlines()]

    header=blocks[0]

    blocks_nparray = np.array(blocks[1:], dtype=np.float64)

    labels = [sanitize_header_label(label) for label in header[1:]]

    for i_op in range(1, len(header)-1, 2):
        label = labels[i_op]

        trees = blocks_nparray[:,0]
        train_losses = blocks_nparray[:,i_op]
        valid_losses = blocks_nparray[:,i_op+1]

        plt.figure(i_op)
        plt.plot(trees, train_losses, label="train")
        plt.plot(trees, valid_losses, label="valid")
        plt.xlabel("n_trees")
        plt.ylabel("ratio_mse_loss")
        plt.axvline(np.argmin(valid_losses), color='r', label="best epoch")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.legend(title=label)

        loss_pdf = os.path.join(plot_dir, f"loss_history_{label}.pdf")
        plt.tight_layout()
        plt.savefig(loss_pdf, dpi=500)
        plt.close()            

def plot_bit_losses(plot_dir, loss_txt):

    blocks = []
    with open(loss_txt, "r") as f:
        blocks = [line.strip().split("\t") for line in f.readlines()]

    blocks_nparray = np.array(blocks[1:], dtype=np.float64)

    trees = blocks_nparray[:,0]
    train_losses = blocks_nparray[:,1]
    valid_losses = blocks_nparray[:,2]

    plt.plot(trees, train_losses, label="train")
    if np.isfinite(valid_losses).any():
        plt.plot(trees, valid_losses, label="valid")
    plt.xlabel("n_trees")
    plt.ylabel("ratio_mse_loss")
    plt.axvline(np.argmin(valid_losses), color='r', label="best epoch")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()

    loss_pdf = os.path.join(plot_dir, f"loss_history.pdf")
    plt.tight_layout()
    plt.savefig(loss_pdf, dpi=500)
    plt.close()



if __name__ == "__main__":

    for path in glob.glob("models_SBIEFT/unbinned_v7_eft_genpoint/SR_2018/BIT/**/loss_history.txt"):
        output_path = os.path.dirname(path).removeprefix("models_SBIEFT").replace("BIT/","")
        logger.info(f"Getting train/val loss averaged over all terms from {output_path}")
        plot_dir = user.plot_directory+"BIT/"+output_path
        plot_bit_losses(plot_dir, path)

    for path in glob.glob("models_SBIEFT/unbinned_v7_eft_genpoint/SR_2018/BIT/**/loss_history_all_terms.txt"):
        output_path = os.path.dirname(path).removeprefix("models_SBIEFT").replace("BIT/","")
        logger.info(f"Getting train/val loss averaged over all terms from {output_path}")
        plot_dir = user.plot_directory+"BIT/"+output_path
        plot_bit_losses_all_terms(plot_dir, path)
