#python ic_training.py --selection inclusive
#python ic_training.py --selection lowMT_VBFJet
#python ic_training.py --selection highMT_VBFJet
python ic_training.py --selection lowMT_noVBFJet_ptH0to100
#python ic_training.py --selection lowMT_noVBFJet_ptH100
#python ic_training.py --selection highMT_noVBFJet

python ic_training.py --selection highMT_noVBFJet --mvaSelection MVAHighMTnoVBFJetTtbar
python ic_training.py --selection highMT_noVBFJet --mvaSelection MVAHighMTnoVBFJetDiboson 

#python ic_training.py --selection highMT
#python ic_training.py --overwrite --selection GGHMVA_lowMT_noVBFJet_ptH0to100
