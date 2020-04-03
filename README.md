# Adversarial Validation Approach to Concept Drift Problem in Automated Machine Learning Systems

This repository provides code for the paper titled "Adversarial Validation Approach to Concept Drift Problem in Automated Machine Learning Systems". The code is based on the AutoML3 starting kit.

The original AutoML3 starting kit and public input data distributed by the organizers are available at [the AutoML3 competition site](https://competitions.codalab.org/competitions/19836#participate).

# Adversarial Validation Methods

Three different adversarial validation methods are implemented: feature selection, validation selection and inverse propensity weighting. For the feature selection method, three algorithms of GBDT, Decision Trees, and Random Forests are used in adversarial classifier model training.

Model code for each method is available as follows:
* `model/baseline/`: The baseline model without adversarial validation
* `model/av_fs_dt/`: The adversarial validation with feature selection and decision tree adversarial classifier
* `model/av_fs_rf/`: The adversarial validation with feature selection and random forests classifier
* `model/av_fs_lgb/`: The adversarial validation with feature selection and gradient boosted decision trees classifier
* `model/av_val/`: The adversarial validation with validation selection
* `model/av_ipw/`: The adversarial validation with inverse propensity weighting

# How to Run

You can run the data ingestion, model training and scoring as follows:
```bash
$ ./run.sh [model folder] [data folder]
```
For example, to run the starting kit with sample model code and data, run `run.sh` as follows:
```bash
$ ./run.sh model/baseline AutoML3_sample_data
```

The output files including predictions and scores are saved in `build/[model name]`.

For example, the output files from the run with the sample model code and data will be available as follows:

```bash
$ ls -alF ./build/AutoML3_sample_code_submission/
total 2952
drwxr-xr-x  11 jeong  staff   352B Feb 25 17:00 ./
drwxr-xr-x   4 jeong  staff   128B Feb 25 16:59 ../
-rw-r--r--   1 jeong  staff   414K Feb 25 17:00 ada_test1.predict
-rw-r--r--   1 jeong  staff   415K Feb 25 17:00 ada_test2.predict
-rw-r--r--   1 jeong  staff   416K Feb 25 17:00 ada_test3.predict
drwxr-xr-x   3 jeong  staff    96B Feb 25 17:00 res/
-rw-r--r--   1 jeong  staff    44K Feb 25 17:00 rl_test1.predict
-rw-r--r--   1 jeong  staff    43K Feb 25 17:00 rl_test2.predict
-rw-r--r--   1 jeong  staff   128K Feb 25 17:00 rl_test3.predict
-rw-r--r--   1 jeong  staff   2.7K Feb 25 17:00 scores.html
-rw-r--r--   1 jeong  staff    73B Feb 25 17:00 scores.txt
```
