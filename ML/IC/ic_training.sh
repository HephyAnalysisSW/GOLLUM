#python ic_training.py --selection inclusive
#python ic_training.py --selection lowMT_VBFJet
#python ic_training.py --selection highMT_VBFJet
#python ic_training.py --selection lowMT_noVBFJet_ptH0to100
#python ic_training.py --selection lowMT_noVBFJet_ptH100
#python ic_training.py --selection highMT_noVBFJet

#python ic_training.py --selection highMT_noVBFJet --mvaSelection MVAHighMTnoVBFJetTtbar
#python ic_training.py --selection highMT_noVBFJet --mvaSelection MVAHighMTnoVBFJetDiboson 

#python ic_training.py --selection highMT
#python ic_training.py --overwrite --selection GGHMVA_lowMT_noVBFJet_ptH0to100


python ic_training.py --selection lowMT_VBFJet --mvaSelection    MVALowMTVBFJet_bin0
python ic_training.py --selection lowMT_VBFJet --mvaSelection    MVALowMTVBFJet_bin1
python ic_training.py --selection lowMT_VBFJet --mvaSelection    MVALowMTVBFJet_bin2
python ic_training.py --selection lowMT_VBFJet --mvaSelection    MVALowMTVBFJet_bin3
python ic_training.py --selection lowMT_VBFJet --mvaSelection    MVALowMTVBFJet_bin4
python ic_training.py --selection lowMT_VBFJet --mvaSelection    MVALowMTVBFJet_bin5
python ic_training.py --selection lowMT_VBFJet --mvaSelection    MVALowMTVBFJet_bin6

python ic_training.py --selection lowMT_noVBFJet_ptH100 --mvaSelection    MVALowMTNoVBFJetPtH100_bin0
python ic_training.py --selection lowMT_noVBFJet_ptH100 --mvaSelection    MVALowMTNoVBFJetPtH100_bin1
python ic_training.py --selection lowMT_noVBFJet_ptH100 --mvaSelection    MVALowMTNoVBFJetPtH100_bin2
python ic_training.py --selection lowMT_noVBFJet_ptH100 --mvaSelection    MVALowMTNoVBFJetPtH100_bin3
python ic_training.py --selection lowMT_noVBFJet_ptH100 --mvaSelection    MVALowMTNoVBFJetPtH100_bin4

python ic_training.py --selection highMT_VBFJet --mvaSelection    MVAHighMTVBJet_bin0
python ic_training.py --selection highMT_VBFJet --mvaSelection    MVAHighMTVBJet_bin1
python ic_training.py --selection highMT_VBFJet --mvaSelection    MVAHighMTVBJet_bin2
python ic_training.py --selection highMT_VBFJet --mvaSelection    MVAHighMTVBJet_bin3
python ic_training.py --selection highMT_VBFJet --mvaSelection    MVAHighMTVBJet_bin4
