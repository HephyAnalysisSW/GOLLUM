python pnn_training.py --selection lowMT_VBFJet --config pnn_quad_jes_v2 --training v4 --every 1
python pnn_training.py --selection lowMT_noVBFJet_ptH0to100 --config pnn_quad_jes_v2 --n_split 50 --training v4 --every 1
python pnn_training.py --selection lowMT_noVBFJet_ptH100 --config pnn_quad_jes_v2 --training v4 --every 1
python pnn_training.py --selection highMT_VBFJet --config pnn_quad_jes_v2 --training v4 --every 1
python pnn_training.py --selection highMT_noVBFJet --config pnn_quad_jes_v2 --training v4 --every 1

python pnn_training.py --selection lowMT_VBFJet --config pnn_quad_tes_v2 --training v5 --every 1
python pnn_training.py --selection lowMT_noVBFJet_ptH0to100 --config pnn_quad_tes_v2 --n_split 50 --training v5 --every 1
python pnn_training.py --selection lowMT_noVBFJet_ptH100 --config pnn_quad_tes_v2 --training v5 --every 1
python pnn_training.py --selection highMT_VBFJet --config pnn_quad_tes_v2 --training v5 --every 1
python pnn_training.py --selection highMT_noVBFJet --config pnn_quad_tes_v2 --training v5 --every 1
