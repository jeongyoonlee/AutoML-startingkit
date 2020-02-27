# AutoML3 Starting Kit

This repository provides a revised version of AutoML3 starting kit.

The original AutoML3 starting kit and public input data distributed by the organizers are available at [the AutoML3 competition site](https://competitions.codalab.org/competitions/19836#participate).

# How to Run

You can run the data ingestion, model training and scoring as follows:
```bash
$ ./run.sh [model folder] [data folder]
```
For example, to run the starting kit with sample model code and data, run `run.sh` as follows:
```bash
$ ./run.sh model/AutoML3_sample_code_submission AutoML3_sample_data
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
