# Do this first
#python runInference.py --config config_reference_xgbmc.yaml --save --overwrite --logLevel DEBUG --modify CSI.save=False

# Then do these (parallel)
python runInference.py --config config_reference.yaml --save --logLevel DEBUG --modify CSI.save=True --CSI htautau lowMT_VBFJet 
python runInference.py --config config_reference.yaml --save --logLevel DEBUG --modify CSI.save=True --CSI ztautau lowMT_VBFJet 
python runInference.py --config config_reference.yaml --save --logLevel DEBUG --modify CSI.save=True --CSI ttbar   lowMT_VBFJet 
python runInference.py --config config_reference.yaml --save --logLevel DEBUG --modify CSI.save=True --CSI diboson lowMT_VBFJet 

python runInference.py --config config_reference.yaml --save --logLevel DEBUG --modify CSI.save=True --CSI htautau lowMT_noVBFJet_ptH100 
python runInference.py --config config_reference.yaml --save --logLevel DEBUG --modify CSI.save=True --CSI ztautau lowMT_noVBFJet_ptH100 
python runInference.py --config config_reference.yaml --save --logLevel DEBUG --modify CSI.save=True --CSI ttbar   lowMT_noVBFJet_ptH100 
python runInference.py --config config_reference.yaml --save --logLevel DEBUG --modify CSI.save=True --CSI diboson lowMT_noVBFJet_ptH100 

python runInference.py --config config_reference.yaml --save --logLevel DEBUG --modify CSI.save=True --CSI htautau highMT_VBFJet 
python runInference.py --config config_reference.yaml --save --logLevel DEBUG --modify CSI.save=True --CSI ztautau highMT_VBFJet 
python runInference.py --config config_reference.yaml --save --logLevel DEBUG --modify CSI.save=True --CSI ttbar   highMT_VBFJet 
python runInference.py --config config_reference.yaml --save --logLevel DEBUG --modify CSI.save=True --CSI diboson highMT_VBFJet 

python runInference.py --config config_reference_xgbmc.yaml --save --logLevel DEBUG --modify CSI.save=True --CSI htautau lowMT_VBFJet 
python runInference.py --config config_reference_xgbmc.yaml --save --logLevel DEBUG --modify CSI.save=True --CSI ztautau lowMT_VBFJet 
python runInference.py --config config_reference_xgbmc.yaml --save --logLevel DEBUG --modify CSI.save=True --CSI ttbar   lowMT_VBFJet 
python runInference.py --config config_reference_xgbmc.yaml --save --logLevel DEBUG --modify CSI.save=True --CSI diboson lowMT_VBFJet 

python runInference.py --config config_reference_xgbmc.yaml --save --logLevel DEBUG --modify CSI.save=True --CSI htautau lowMT_noVBFJet_ptH100 
python runInference.py --config config_reference_xgbmc.yaml --save --logLevel DEBUG --modify CSI.save=True --CSI ztautau lowMT_noVBFJet_ptH100 
python runInference.py --config config_reference_xgbmc.yaml --save --logLevel DEBUG --modify CSI.save=True --CSI ttbar   lowMT_noVBFJet_ptH100 
python runInference.py --config config_reference_xgbmc.yaml --save --logLevel DEBUG --modify CSI.save=True --CSI diboson lowMT_noVBFJet_ptH100 

python runInference.py --config config_reference_xgbmc.yaml --save --logLevel DEBUG --modify CSI.save=True --CSI htautau highMT_VBFJet 
python runInference.py --config config_reference_xgbmc.yaml --save --logLevel DEBUG --modify CSI.save=True --CSI ztautau highMT_VBFJet 
python runInference.py --config config_reference_xgbmc.yaml --save --logLevel DEBUG --modify CSI.save=True --CSI ttbar   highMT_VBFJet 
python runInference.py --config config_reference_xgbmc.yaml --save --logLevel DEBUG --modify CSI.save=True --CSI diboson highMT_VBFJet 

#python runInference.py --config config_reference.yaml --save --logLevel DEBUG --modify CSI.save=True --CSI htautau  
#python runInference.py --config config_reference.yaml --save --logLevel DEBUG --modify CSI.save=True --CSI ztautau  
#python runInference.py --config config_reference.yaml --save --logLevel DEBUG --modify CSI.save=True --CSI ttbar    
#python runInference.py --config config_reference.yaml --save --logLevel DEBUG --modify CSI.save=True --CSI diboson  
#
#python runInference.py --config config_reference_xgbmc.yaml --save --logLevel DEBUG --modify CSI.save=True --CSI htautau  
#python runInference.py --config config_reference_xgbmc.yaml --save --logLevel DEBUG --modify CSI.save=True --CSI ztautau  
#python runInference.py --config config_reference_xgbmc.yaml --save --logLevel DEBUG --modify CSI.save=True --CSI ttbar    
#python runInference.py --config config_reference_xgbmc.yaml --save --logLevel DEBUG --modify CSI.save=True --CSI diboson  
