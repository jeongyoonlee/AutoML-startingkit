# `av_fs`: An adversarial validation with feature selection

This model is based on the `baseline_fe` model, which uses the setup as follows:

* At each batch, it trains a model only with new training data without using previous training data
* At each batch, it uses 100% of training data
* It uses LightGBM's LGBMClassifier
* It frequency-encodes categorical features

We added an adversarial validation to `baseline_fe` as follows:

* At each batch, it trains an adversarial validation model with the training and test feature data
* At each batch, it drops up to 5 features with highest feature importances that are greater than 0.1 Gini score.
