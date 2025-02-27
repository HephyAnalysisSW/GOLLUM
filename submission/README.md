# Submission

## Instructions:

Please make sure running the commands under the main folder: `/path/to/HEPHY-uncertainty`

Prepare the submission tarball: `python submission/prepareSubmission.py -c Workflow/configs/config_reference_v2_calib.yaml --ntuple /groups/hephy/cms/dennis.schwarz/HiggsChallenge/output/config_reference_v2_calib/tmp_data`. This copies all of the used models (only the latest checkpoint), calibrations, icp files into a local models directory. The CSI pickle files are copied to a local data directory. Then the whole thing is tarred together and saved as `../submission.tar`.

Run an example fit locally for sanity check: `python submission/runModel_internal_test.py`

If we want, we can submit the `submission.tar`.
