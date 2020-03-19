'''
Sample predictive model.
You must supply at least 4 methods:
- fit: trains the model.
- predict: uses the model to perform predictions.
- save: saves the model.
- load: reloads the model.
PLEASE NOTE THAT WE ARE PASSING THE INFO OF THE DATA SET AS AN ADDITIONAL ARGUMENT!
'''
import pickle
import data_converter
import numpy as np   # We recommend to use numpy arrays
from os.path import isfile
import time
from causalml.propensity import calibrate
from lightgbm import LGBMClassifier
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from kaggler.preprocessing import FrequencyEncoder


SEED = 42
GINI_THRESHOLD = .1


class Model:
    def __init__(self,datainfo,timeinfo):
        '''
        This constructor is supposed to initialize data members.
        Use triple quotes for function documentation.
        '''
        # Just print some info from the datainfo variable
        print("The Budget for this data set is: %d seconds" %datainfo['time_budget'])

        print("Loaded %d time features, %d numerical Features, %d categorical features and %d multi valued categorical variables" %(datainfo['loaded_feat_types'][0], datainfo['loaded_feat_types'][1],datainfo['loaded_feat_types'][2],datainfo['loaded_feat_types'][3]))
        overall_spenttime=time.time()-timeinfo[0]
        dataset_spenttime=time.time()-timeinfo[1]
        print("[***] Overall time spent %5.2f sec" % overall_spenttime)
        print("[***] Dataset time spent %5.2f sec" % dataset_spenttime)
        self.num_train_samples=0
        self.num_feat=1
        self.num_labels=1
        self.is_trained=False
        self.clf = LGBMClassifier(n_estimators=1000, subsample=.8, subsample_freq=1, colsample_bytree=.8, importance_type='gain')
        # Here you may have parameters and hyper-parameters

    def fit(self, F, y, datainfo,timeinfo):
        '''
        This function should train the model parameters.
        Here we do nothing in this example...
        Args:
            X: Training data matrix of dim num_train_samples * num_feat.
            y: Training label matrix of dim num_train_samples * num_labels.
        Both inputs are numpy arrays.
        If fit is called multiple times on incremental data (train, test1, test2, etc.)
        you should warm-start your training from the pre-trained model. Past data will
        NOT be available for re-training.
        '''

        overall_spenttime=time.time()-timeinfo[0]
        dataset_spenttime=time.time()-timeinfo[1]

        print("[***] Overall time spent %5.2f sec" % overall_spenttime)
        print("[***] Dataset time spent %5.2f sec" % dataset_spenttime)

        date_cols = datainfo['loaded_feat_types'][0]
        numeric_cols = datainfo['loaded_feat_types'][1]
        categorical_cols = datainfo['loaded_feat_types'][2]
        multicategorical_cols = datainfo['loaded_feat_types'][3]

        # Get numerical variables and replace NaNs with 0s
        self.X = np.nan_to_num(F['numerical'])
        self.y = y

        # Frequency encode categorical variables and concatenate them with numerical variables
        if categorical_cols > 0:
            self.cat_encs = FrequencyEncoder()
            X_cat = self.cat_encs.fit_transform(F['CAT']).values
            self.X = np.concatenate((self.X, X_cat), axis=1)
            del X_cat

        self.num_train_samples = self.X.shape[0]
        self.num_feat = self.X.shape[1]
        num_train_samples = y.shape[0]

        print ("The whole available data is: ")
        print(("Real-FIT: dim(X)= [{:d}, {:d}]").format(self.X.shape[0],self.X.shape[1]))
        print(("Real-FIT: dim(y)= [{:d}, {:d}]").format(self.y.shape[0], self.num_labels))

        self.is_trained=True

    def predict(self, F,datainfo,timeinfo):
        '''
        This function should provide predictions of labels on (test) data.
        Here we just return random values...
        Make sure that the predicted values are in the correct format for the scoring
        metric. For example, binary classification problems often expect predictions
        in the form of a discriminant value (if the area under the ROC curve it the metric)
        rather that predictions of the class labels themselves.
        The function predict eventually casdn return probabilities or continuous values.
        '''

        overall_spenttime=time.time()-timeinfo[0]
        dataset_spenttime=time.time()-timeinfo[1]

        print("[***] Overall time spent %5.2f sec" % overall_spenttime)
        print("[***] Dataset time spent %5.2f sec" % dataset_spenttime)

        date_cols = datainfo['loaded_feat_types'][0]
        numeric_cols = datainfo['loaded_feat_types'][1]
        categorical_cols = datainfo['loaded_feat_types'][2]
        multicategorical_cols = datainfo['loaded_feat_types'][3]

        # Get numerical variables and replace NaNs with 0s
        X = np.nan_to_num(F['numerical'])

        # Frequency encode categorical variables and concatenate them with numerical variables
        if categorical_cols > 0:
            X_cat = self.cat_encs.transform(F['CAT']).values
            X = np.concatenate((X, X_cat), axis=1)
            del X_cat

        # Adversarial validation
        print('AV: starting adversarial validation...')

        np.random.seed(SEED)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

        n_trn = self.X.shape[0]
        n_tst = X.shape[0]

        X_all = np.vstack((self.X, X))
        y_all = np.concatenate((np.zeros(n_trn,), np.ones(n_tst,)))
        print('AV: ', X_all.shape, y_all.shape)

        ps_all = np.zeros_like(y_all, dtype=float)
        for i, (i_trn, i_val) in enumerate(cv.split(X_all, y_all)):

            model_av = LGBMClassifier(n_estimators=1000, subsample=.8, subsample_freq=1, colsample_bytree=.8, importance_type='gain')
            model_av.fit(X_all[i_trn], y_all[i_trn],
                        eval_set=(X_all[i_val], y_all[i_val]),
                        early_stopping_rounds=10,
                        eval_metric='auc')

            ps_all[i_val] = model_av.predict_proba(X_all[i_val])[:, 1]

        av_score = roc_auc_score(y_all, ps_all)
        print(f'AV: AUC={av_score * 100: 3.2f}')

        ps_all = np.clip(calibrate(ps_all, y_all), .1, .9)
        w_all = ps_all / (1 - ps_all)
        print(f'AV: propensity scores deciles: {np.percentile(ps_all, np.linspace(0, 1, 11))}')

        # Training
        X_trn, X_val, y_trn, y_val, w_trn, w_val = train_test_split(self.X, self.y, w_all[:n_trn], test_size=.25, random_state=SEED)
        self.clf.fit(X_trn, y_trn,
                     eval_set=(X_val, y_val),
                     early_stopping_rounds=10,
                     eval_metric='auc',
                     sample_weight=w_trn)

        num_test_samples = X.shape[0]
        if X.ndim > 1: num_feat = X.shape[1]
        print(("PREDICT: dim(X)= [{:d}, {:d}]").format(num_test_samples, num_feat))
        if (self.num_feat != num_feat):
            print("ARRGH: number of features in X does not match training data!")
        print(("PREDICT: dim(y)= [{:d}, {:d}]").format(num_test_samples, self.num_labels))
        y= self.clf.predict_proba(X)[:, 1]
        y= np.transpose(y)
        return y

    def save(self, path="./"):
        pickle.dump(self, open(path + '_model.pickle', "w"))

    def load(self, path="./"):
        modelfile = path + '_model.pickle'
        if isfile(modelfile):
            with open(modelfile) as f:
                self = pickle.load(f)
            print("Model reloaded from: " + modelfile)
        return self
