# `baseline_fe`: A baseline model with frequency encoding of categorical features

This model is based on the `baseline` model, which uses the setup as follows:

* At each batch, it trains a model only with new training data without using previous training data
* At each batch, it uses 100% of training data
* It uses LightGBM's LGBMClassifier

We added an update to `baseline` as follows:

* It frequency-encodes categorical features
