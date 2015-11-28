__author__ = 'davidnola'

import numpy as np
from nolearn.lasagne import NeuralNet
from sklearn.base import BaseEstimator, TransformerMixin,ClassifierMixin
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import pickle
import sys
import sklearn.metrics
import lasagne.nonlinearities



# Neural Net for 1d samples and categorical labels:
class PipelineNet(NeuralNet): # By default Lasagne is super finicky with inputs and outputs. So I just handle most of the pre and postprocessing for you.
    def fit(self,X, y,**params):
        self.label_encoder = LabelEncoder()
        self.one_hot = OneHotEncoder()

        y = list(map(lambda x:[x],self.label_encoder.fit_transform(y)))
        y = np.array(self.one_hot.fit_transform(y).toarray(),dtype=np.float32)
        X = np.array(X,dtype=np.float32)

        self.output_num_units=len(y[0])
        self.input_shape=(None,X.shape[1])

        self.output_nonlinearity=lasagne.nonlinearities.softmax

        return NeuralNet.fit(self,X,y,**params)

    def predict(self, X):
        X = np.array(X,dtype=np.float32)
        preds = NeuralNet.predict(self,X)

        preds = np.argmax(preds,axis=1)
        preds = self.label_encoder.inverse_transform(preds)

        return preds

    def score(self, X, y):
        return sklearn.metrics.accuracy_score(self.predict(X),y)


class EarlyStopping(object): # I stole this from Nouri's tutorial - Nouri is the guy who made lasagne and nolearn. He really should have included this code by default.
    def __init__(self, patience=50):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = nn.get_all_params_values()
        elif self.best_valid_epoch + self.patience < current_epoch:
            print("Early stopping.")
            print("Best valid loss was {:.6f} at epoch {}.".format(
                self.best_valid, self.best_valid_epoch))
            nn.load_params_from(self.best_weights)
            raise StopIteration()


def float32(k):
    return np.cast['float32'](k)