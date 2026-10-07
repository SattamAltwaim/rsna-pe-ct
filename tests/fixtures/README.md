# Test fixtures

`bc855cd8bdc9_8slices.tar.gz` holds 8 consecutive DICOM slices (instance numbers
21-28) of study `bc855cd8bdc9` from the RSNA-STR Pulmonary Embolism CT dataset,
3 of which are labelled PE-positive. `bc855cd8bdc9_labels.csv` holds the matching
rows of `train.csv`, and `bc855cd8bdc9_zip_index.csv` the matching rows of the zip
index (byte offsets inside the public zip), used by the optional network test.

The data is de-identified and redistributed here, for non-commercial research and
testing only, under the dataset's terms, which require the following citation:

> RSNA-STR Pulmonary Embolism CT (RSPECT) Dataset, Copyright RSNA, 2020.
> https://registry.opendata.aws/rsna-pulmonary-embolism-detection

Everything else in the tests is synthetic (generated with pydicom / numpy).
