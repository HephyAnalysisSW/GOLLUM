## Prediction:

To predict the signal strength interval:

```
python predict.py --TEST_DATA_CLEAN_PATH example_test_data/example_test_data.h5 --SUBMISSION_DIR submission
```

Due to the assumption that requires non-zero events for each selection, the `--nevents` arguments are NOT provided to avoid runtime issues.
