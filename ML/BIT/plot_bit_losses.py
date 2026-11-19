import numpy as np
import common.user as user
import common.syncer as syncer
import common.yaml_loader as yaml_loader 
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
    

def plot_bit_losses_all_terms(plot_dir: str, loss_txt_all_terms: str, best_epoch_average: int):

    header = []
    blocks = []
    with open(loss_txt_all_terms, "r") as f:
        blocks = [line.strip().split("\t") for line in f.readlines()]

    header=blocks[0]

    blocks_nparray = np.array(blocks[1:], dtype=np.float64)

    sanitized_header = [sanitize_header_label(label) for label in header]

    for i_op in range(2, len(header)):
        label = sanitized_header[i_op]
        trees = blocks_nparray[:,0]
        train_losses = blocks_nparray[:,1]
        #logger.info(train_losses)
        valid_losses = blocks_nparray[:,i_op]

        plt.figure(i_op)
        plt.plot(trees, train_losses, label="train")
        plt.plot(trees, valid_losses, label="valid")
        plt.xlabel("n_trees")
        plt.ylabel("ratio_mse_loss")
        plt.axvline(np.argmin(valid_losses), color='r', label="best epoch (term)")
        if best_epoch_average:
            plt.axvline(best_epoch_average, color='g', label="best epoch (overall)")
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
    best_epoch = np.argmin(valid_losses) 
    plt.axvline(best_epoch, color='r', label="best epoch")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()

    loss_pdf = os.path.join(plot_dir, f"loss_history.pdf")
    plt.tight_layout()
    plt.savefig(loss_pdf, dpi=500)
    plt.close()

    return best_epoch


if __name__ == "__main__":

    parser = ap.ArgumentParser(description="plots single and per-term loss for a BIT training job")
    parser.add_argument("config")
    parser.add_argument("--job")
    
    args = parser.parse_args()

    cfg = yaml_loader.load_yaml(args.config)

    # ---------------- list mode ----------------
    if args.job is None:
        jobs = [j for j in (cfg.get("jobs") or []) if j.get("type") == "bit"]
        if not jobs:
            print("No BIT jobs found in YAML.")
            sys.exit(0)
        #script = os.path.basename(__file__)
        for j in jobs:
            print(f"python {__file__} {args.config} --job {j['id']}".strip())
        sys.exit(0)


    # ---------------- resolve job ----------------
    job = next((j for j in (cfg.get("jobs") or []) if j.get("id") == args.job), None)
    if job is None:
        raise RuntimeError(f"Job id '{args.job}' not found.")
    if job.get("type") != "bit":
        raise RuntimeError(f"Job '{args.job}' is not a BIT job.")

    model_dir = os.path.join(user.model_directory, cfg.get("version"), job["region"], "BIT", job["id"])
    plot_dir = os.path.join(user.plot_directory,"BIT", cfg.get("version"), job["region"], job["id"])
    
    n_ensemble = job.get("n_ensemble")
    logger.info(f"plotting losses for {job['id']=}")
    if n_ensemble:
        for i_ensemble in range(n_ensemble):
            print(f"{i_ensemble=}")
            best_epoch_average = plot_bit_losses(os.path.join(plot_dir,f"ensemble_{i_ensemble}"),
                            os.path.join(model_dir,f"ensemble_{i_ensemble}","loss_history.txt")) 
            plot_bit_losses_all_terms(os.path.join(plot_dir,f"ensemble_{i_ensemble}"),
                            os.path.join(model_dir,f"ensemble_{i_ensemble}","loss_history_all_terms.txt"), best_epoch_average=best_epoch_average)
    else:
        best_epoch_average = plot_bit_losses(plot_dir,
                        os.path.join(model_dir,"loss_history.txt")) 
        plot_bit_losses_all_terms(plot_dir,
                        os.path.join(model_dir,"loss_history_all_terms.txt"), best_epoch_average=best_epoch_average)        