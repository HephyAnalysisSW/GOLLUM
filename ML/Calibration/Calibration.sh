#python Calibration.py --config config_reference_v2_sr --selection lowMT_VBFJet
#python Calibration.py --config config_reference_v2_calib --selection lowMT_VBFJet
#python Calibration.py --config config_reference_v2_calib --selection lowMT_noVBFJet_ptH100
#python Calibration.py --config config_reference_v2_calib --selection highMT_VBFJet

python Calibration.py --config config_reference_v3 --selection lowMT_VBFJet
python Calibration.py --config config_reference_v3 --selection lowMT_noVBFJet_ptH100
python Calibration.py --config config_reference_v3 --selection highMT_VBFJet

