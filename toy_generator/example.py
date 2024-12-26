from toy_generator import run_pseudo_experiments

run_pseudo_experiments(
    tes=1.05,
    jes=0.97,
    soft_met=0.8,
    ttbar_scale=None,
    diboson_scale=1.15,
    bkg_scale=None,
    num_pseudo_experiments=2,
    num_of_sets=3,
    ground_truth_mus=[1.0, 1.1, 1.2],
    output_dir="/eos/vbc/group/mlearning/data/Higgs_uncertainty/input_data/split_train_dataset/pseudo/",
)
