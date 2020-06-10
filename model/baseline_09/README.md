# `baseline`: A baseline model with label encoding of categorical features

This model is based on the AutoML3 sample model, which uses the setup as follows:

* At each batch, it trains a model only with new training data without using previous training data
* At each batch, it uses 100% of training data
* It uses scikit-learn's GradientBoostingClassifier

We added an update to `baseline` as follows:

* It uses LightGBM's LGBMClassifier
* It label-encodes categorical features with missing values as a new label