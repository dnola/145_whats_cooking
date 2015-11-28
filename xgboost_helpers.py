from sklearn.base import BaseEstimator, TransformerMixin,ClassifierMixin
from sklearn.externals.six import iteritems
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import numpy as np
import sklearn.cluster
import json
import sklearn.feature_selection
import random
import xgboost as xgb

class XGBClassifier(BaseEstimator,ClassifierMixin):
    def __init__(self,maxdepth=20,objective="multi:softmax",):
        self.params = {'max.depth':20,'objective':"multi:softmax", 'num_class': len(np.unique(y))}


    def fit(self,X,y=None):
        self.params['num_class'] = len(np.unique(y))
        self.encoder=LabelEncoder()
        y = self.encoder.fit_transform(y)
        data = xgb.DMatrix( np.array(X), label=y)
        self.model = xgb.train(self.params,data,10)
        return self

    def predict(self,X):
        data = xgb.DMatrix( np.array(X))
        preds = self.model.predict(data)
        print(preds[:10])
        preds = self.encoder.inverse_transform(preds)
        print(preds[:10])
        return preds
