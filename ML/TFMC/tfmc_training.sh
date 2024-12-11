#python tfmc_training.py --selection lowMT_noVBFJet_pt0to100 --config tfmc_1 --n_split 50 
#python tfmc_training.py --selection lowMT_noVBFJet_pt0to100 --config tfmc_1 

python tfmc_training.py --config tfmc_2_reg --selection  lowMT_VBFJet
python tfmc_training.py --config tfmc_2_reg --selection  highMT_VBFJet
#python tfmc_training.py --config tfmc_2_reg --selection  lowMT_noVBFJet_ptH0to100 --n_split 50
python tfmc_training.py --config tfmc_2_reg --selection  lowMT_noVBFJet_ptH100
python tfmc_training.py --config tfmc_2_reg --selection  highMT_noVBFJet_ptH0to100
python tfmc_training.py --config tfmc_2_reg --selection  highMT_noVBFJet_ptH100

