#python plot_systematics.py  --config pnn_quad_jes_all --selection lowMT_VBFJet
#python plot_systematics.py  --config pnn_quad_jes_all --selection highMT_VBFJet
python plot_systematics.py  --config pnn_quad_jes_all --selection lowMT_noVBFJet_ptH0to100
python plot_systematics.py  --config pnn_quad_jes_all --selection lowMT_noVBFJet_ptH100
python plot_systematics.py  --config pnn_quad_jes_all --selection highMT_noVBFJet_ptH0to100
python plot_systematics.py  --config pnn_quad_jes_all --selection highMT_noVBFJet_ptH100

#python plot_systematics.py  --config pnn_quad_tes_all --selection lowMT_VBFJet
#python plot_systematics.py  --config pnn_quad_tes_all --selection highMT_VBFJet
python plot_systematics.py  --config pnn_quad_tes_all --selection lowMT_noVBFJet_ptH0to100
python plot_systematics.py  --config pnn_quad_tes_all --selection lowMT_noVBFJet_ptH100
python plot_systematics.py  --config pnn_quad_tes_all --selection highMT_noVBFJet_ptH0to100
python plot_systematics.py  --config pnn_quad_tes_all --selection highMT_noVBFJet_ptH100

python plot_systematics.py  --config pnn_quad_met --selection lowMT_VBFJet
python plot_systematics.py  --config pnn_quad_met --selection highMT_VBFJet
python plot_systematics.py  --config pnn_quad_met --selection lowMT_noVBFJet_ptH0to100
python plot_systematics.py  --config pnn_quad_met --selection lowMT_noVBFJet_ptH100
python plot_systematics.py  --config pnn_quad_met --selection highMT_noVBFJet_ptH0to100
python plot_systematics.py  --config pnn_quad_met --selection highMT_noVBFJet_ptH100
