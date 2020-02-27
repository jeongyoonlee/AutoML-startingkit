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
import random
import time
from sklearn.ensemble import GradientBoostingClassifier


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
        self.clf = GradientBoostingClassifier(n_estimators=5, verbose=1, random_state=1, min_samples_split=10, warm_start = False)
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
        # get the raw categorical and categorical multivalued variables in case you want to process them, in this baseline we simply ignore them
        MV=F['MV']
        CAT=F['CAT']

        # only get numerical variables
        X=F['numerical']

        overall_spenttime=time.time()-timeinfo[0]
        dataset_spenttime=time.time()-timeinfo[1]

        print("[***] Overall time spent %5.2f sec" % overall_spenttime)
        print("[***] Dataset time spent %5.2f sec" % dataset_spenttime)

        # get numerical variables, concatenate them with categorical variables
        # catnumeric_dataset=np.array(CAT)
        # X= np.concatenate((F['numerical'],catnumeric_dataset),axis=1).astype(np.float64).copy(order='C')

        # convert NaN to zeros
        X = data_converter.replace_missing(X)
        #print "This batch of data has: "
        self.num_train_samples = X.shape[0]
        if X.ndim>1: self.num_feat = X.shape[1]
        #print("FIT: dim(X)= [{:d}, {:d}]").format(self.num_train_samples, self.num_feat)
        num_train_samples = y.shape[0]
        if y.ndim>1: self.num_labels = y.shape[1]
        #print("FIT: dim(y)= [{:d}, {:d}]").format(num_train_samples, self.num_labels)
	    # subsample the data for efficient processing
        removeperc=0.9
        if removeperc>0:
            rem_samples=int(num_train_samples*removeperc)
            skip = sorted(random.sample(range(num_train_samples),num_train_samples-rem_samples))
            num_train_samples=num_train_samples-rem_samples

            X = X[skip,:]
            y = y[skip,:]
            self.num_train_samples = X.shape[0]

        if self.is_trained:
            _ = self.clf.set_params(n_estimators=self.clf.n_estimators+1,warm_start=True)
            self.DataX=X
            self.DataY=y
        else:
            self.DataX=X
            self.DataY=y
        print ("The whole available data is: ")
        print(("Real-FIT: dim(X)= [{:d}, {:d}]").format(self.DataX.shape[0],self.DataX.shape[1]))
        print(("Real-FIT: dim(y)= [{:d}, {:d}]").format(self.DataY.shape[0], self.num_labels))
        #print "fitting with ..."
        #print self.clf.n_estimators
        self.clf.fit(self.DataX,np.ravel(self.DataY))

        #print "Model fitted.."
        if (self.num_train_samples != num_train_samples):
            print("ARRGH: number of samples in X and y do not match!")
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
        # get the raw categorical multivalued variables in case you want to process them, in this baseline we simply ignore them
        MV=F['MV']
        CAT=F['CAT']

        overall_spenttime=time.time()-timeinfo[0]
        dataset_spenttime=time.time()-timeinfo[1]

        print("[***] Overall time spent %5.2f sec" % overall_spenttime)
        print("[***] Dataset time spent %5.2f sec" % dataset_spenttime)

        # only get numerical variables
        X=F['numerical']

        # get numerical variables, concatenate them with categorical variables
        # catnumeric_dataset=np.array(CAT)
        # X= np.concatenate((F['numerical'],catnumeric_dataset),axis=1).astype(np.float64).copy(order='C')

        # convert NaN to zeros
        X = data_converter.replace_missing(X)

        num_test_samples = X.shape[0]
        if X.ndim>1: num_feat = X.shape[1]
        print(("PREDICT: dim(X)= [{:d}, {:d}]").format(num_test_samples, num_feat))
        if (self.num_feat != num_feat):
            print("ARRGH: number of features in X does not match training data!")
        print(("PREDICT: dim(y)= [{:d}, {:d}]").format(num_test_samples, self.num_labels))
        y= self.clf.decision_function(X)
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
