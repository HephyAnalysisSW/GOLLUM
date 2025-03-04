# Submission

## Instructions:

Please make sure running the commands under the main folder: `/path/to/HEPHY-uncertainty`

Install PyYAML: `pip install --target=pythonpackages PyYAML`. When doing this, try to use a higher version of `pip` to avoid problems.

Prepare the submission tarball: `python submission/prepareSubmission.py -c Workflow/configs/config_reference_v2_calib.yaml --ntuple /groups/hephy/cms/dennis.schwarz/HiggsChallenge/output/config_reference_v2_calib/tmp_data`. This copies all of the used models (only the latest checkpoint), calibrations, icp files into a local models directory. The CSI pickle files are copied to a local data directory. Then the whole thing is tarred together and saved as `../submission.tar`.

Run an example fit locally for sanity check: `python submission/runModel_internal_test.py`

If we want, we can submit the `submission.tar`.

## Test submission:
This setup should work within our conda environment.

1. Git clone `git clone git@github.com:FAIR-Universe/HEP-Challenge.git`
2. cd into directory `cd HEP-Challenge`
3. create directory for fit output `sample_result_submission`
4. cd into ingestion_program directory `cd ingestion_program`
5. unzip submission `unzip /groups/hephy/cms/dennis.schwarz/submissions_higgs_challenge/submission_TEST.zip`
6. run `python run_ingestion.py`

You can now go to more realistic test settings via:

`python run_ingestion.py --num-of-sets 1 --num-pseudo-experiments 10 --use-random-mus --systematics-tes --systematics-jes --systematics-soft-met --systematics-ttbar-scale --systematics-diboson-scale --systematics-bkg-scale`

This would create 10 toys for 1 randomly sampled value of mu.
The results are then stored in HEP-Challenge/sample_result_submission/result_*.json
The settings for the toy are stored in HEP-Challenge/sample_result_submission/test_settings.json
