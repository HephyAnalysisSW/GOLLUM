# Submission

## Instructions:

Please make sure running the commands under the main folder: `/path/to/HEPHY-uncertainty`

Prepare the submission tarball: `python submission/prepareSubmission.py -c Workflow/configs/config_reference_v2_calib.yaml --ntuple /groups/hephy/cms/dennis.schwarz/HiggsChallenge/output/config_reference_v2_calib/tmp_data`. This copies all of the used models (only the latest checkpoint), calibrations, icp files into a local models directory. The CSI pickle files are copied to a local data directory. Then the whole thing is tarred together and saved as `../submission.tar`.

Run an example fit locally for sanity check: `python submission/runModel_internal_test.py`

If we want, we can submit the `submission.tar`.

## Test submission:

1. Git clone `git clone git@github.com:FAIR-Universe/HEP-Challenge.git`
2. cd into directory `cd HEP-Challenge`
3. create directory for fit output `sample_result_submission`
4. cd into ingestion_program directory `cd ingestion_program`
5. unzip submission `unzip /groups/hephy/cms/dennis.schwarz/submissions_higgs_challenge/submission_TEST.zip`
6. run `python run_ingestion.py`
