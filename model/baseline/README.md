# `baseline`: A baseline model

This baseline model is based on `AutoML3_sample_code_submission`, which uses the setup as follows:

* At each batch, it trains a model only with new training data without using previous training data

We added an update to `AutoML3_sample_code_submission` as follows:

* It uses LightGBM's LGBMClassifier instead of scikit-learn's GradientBosstingClassifier
* At each batch, it uses 100% of training data instead of 90% random samples

