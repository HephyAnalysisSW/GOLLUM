from toy_generator import run_pseudo_experiments

run_pseudo_experiments(
    tes=None,
    jes=None,
    soft_met=None,
    ttbar_scale=None,
    diboson_scale=None,
    bkg_scale=None,
    num_pseudo_experiments=100,
    num_of_sets=1,
    ground_truth_mus=[1.0],
    output_dir="/eos/vbc/group/mlearning/data/Higgs_uncertainty/input_data/split_train_dataset/processed_data/pseudo_experiments_with_true_labels_mu_1",
)
