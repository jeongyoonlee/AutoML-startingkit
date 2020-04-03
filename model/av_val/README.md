# `av_val`: An adversarial validation with validation data selection

This model is based on the `baseline` model, which uses the setup as follows:

* At each batch, it trains a model only with new training data without using previous training data
* At each batch, it uses 100% of training data
* It uses LightGBM's LGBMClassifier
* It label-encodes categorical features with missing values as a new label

We added an adversarial validation to `baseline` as follows:

* At each batch, it trains an adversarial validation model with the training and test feature data
* At each batch, it uses the subset of training data that ranked high in adversarial validation predictions as the validation set
