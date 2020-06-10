# `av_fs_dt`: An adversarial validation with feature selection and Decision Trees adversarial classifer

This model is based on the `baseline` model, which uses the setup as follows:

* At each batch, it trains a model only with new training data without using previous training data
* At each batch, it uses 100% of training data
* It uses scikit-learn's DecisionTreeClassifier
* It imputes missing values in numerical features with zeros
* It label-encodes categorical features with missing values as a new label

We added an adversarial validation to `baseline` as follows:

* At each batch, it trains an adversarial validation model with the training and test feature data
* At each batch, it keeps dropping top 10% of remaining features with highest feature importances that are greater than 0.1 Gini score until adversarial AUC gets lower than 0.8
