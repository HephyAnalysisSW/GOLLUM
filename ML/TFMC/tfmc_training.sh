
python tfmc_training.py --config tfmc_2_reg --selection  lowMT_VBFJet
python tfmc_training.py --config tfmc_2_reg --selection  highMT_VBFJet
python tfmc_training.py --config tfmc_2_reg --selection  lowMT_noVBFJet_ptH0to100 --n_split 50
python tfmc_training.py --config tfmc_2_reg --selection  lowMT_noVBFJet_ptH100
python tfmc_training.py --config tfmc_2_reg --selection  highMT_noVBFJet

python tfmc_training.py --config tfmc_2_reg_preproc --selection  lowMT_VBFJet
python tfmc_training.py --config tfmc_2_reg_preproc --selection  highMT_VBFJet
python tfmc_training.py --config tfmc_2_reg_preproc --selection  lowMT_noVBFJet_ptH0to100 --n_split 50
python tfmc_training.py --config tfmc_2_reg_preproc --selection  lowMT_noVBFJet_ptH100
python tfmc_training.py --config tfmc_2_reg_preproc --selection  highMT_noVBFJet
