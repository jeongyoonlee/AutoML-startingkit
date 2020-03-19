# `av_p`: An adversarial validation with a propensity feature

This model is based on the `baseline_fe` model, which uses the setup as follows:

* At each batch, it trains a model only with new training data without using previous training data
* At each batch, it uses 100% of training data
* It uses LightGBM's LGBMClassifier
* It frequency-encodes categorical features

We added an adversarial validation to `baseline_fe` as follows:

* At each batch, it trains an adversarial validation model with the training and test feature data
* At each batch, it adds the predictions of the adversarial validation model as a feature to the classifer
