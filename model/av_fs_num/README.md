# `av_fs_num`: An adversarial validation on numerical variables with feature selection

This model is based on the `baseline` model, which uses the setup as follows:

* It uses only numerical features
* At each batch, it trains a model only with new training data without using previous training data
* At each batch, it uses 100% of training data
* It uses LightGBM's LGBMClassifier

We added an adversarial validation to `baseline` as follows:

* At each batch, it trains an adversarial validation model with the training and test feature data
* At each batch, it drops up to top 10% of features with highest feature importances that are greater than 0.1 Gini score.
