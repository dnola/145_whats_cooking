import pickle
import numpy as np
import sklearn.cluster
import sklearn.feature_selection
from sklearn.base import BaseEstimator, TransformerMixin,ClassifierMixin
from sklearn.cross_validation import train_test_split
from sklearn.externals.six import iteritems
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

# Creates a stack ensemble - works similarly to voter ensemble, but takes an additional top level classifier as input
class StackEnsembleClassifier(BaseEstimator,ClassifierMixin):

    def __init__(self, base_classifiers, stack_classifier,hold_out_percent = .90,categorical_labels=True,refit_base=True):
        self.base_classifiers = base_classifiers
        self.stack_classifier = stack_classifier
        self.all_items = base_classifiers+[stack_classifier]
        self.hold_out_percent = hold_out_percent
        self.categorical=categorical_labels
        self.refit_base=refit_base
        if categorical_labels==None:
            self.categorical=True


    def fit(self, X, y=None):
        if self.categorical: # Need to one hot encode labels
            label_encoder = LabelEncoder()
            one_hot = OneHotEncoder()
            label_encoder.fit(y)
            one_hot.fit(list(map(lambda x:[x],label_encoder.transform(y))))
            self.stack_encoder = lambda x: one_hot.transform(list(map(lambda x:[x],label_encoder.transform(x)))).toarray()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-self.hold_out_percent)

        predictions = []
        for (name, clf) in self.base_classifiers:
            print("Ensemble currently fitting:",name)
            clf.fit(X_train,y_train)
            if self.categorical:
                predictions.append(self.stack_encoder(clf.predict(X_test)))
            else:
                predictions.append(list(map(lambda x:[x], clf.predict(X_test))))

        predictions = np.hstack(predictions)

        print("Fitting stack classifier")
        self.stack_classifier[1].fit(predictions,y_test)

        if self.refit_base:
            for (name, clf) in self.base_classifiers:
                print("Ensemble currently refitting:",name)
                clf.fit(X,y)

        return self

    def predict(self, X): # Run prediction and get best models in ensemble
        predictions = []
        for (name, clf) in self.base_classifiers:
            if self.categorical:
                predictions.append(self.stack_encoder(clf.predict(X)))
            else:
                predictions.append(list(map(lambda x:[x], clf.predict(X))))

        predictions = np.hstack(predictions)

        try:
            ranks = np.split(self.stack_classifier[1].ranking_, len(self.base_classifiers))
            ranks = [np.mean(x) for x in ranks]
            print("Ensemble Rankings (Lower is better):", ranks)
        except:
            pass

        return self.stack_classifier[1].predict(predictions)

    def get_params(self, deep=True): # I stole this from the sklearn FeatureUnion code - it basically lets my model work with GridSearch
        if not deep:
            return super(StackEnsembleClassifier, self).get_params(deep=False)
        else:
            out = dict(self.all_items)
            for name, trans in self.all_items:
                for key, value in iteritems(trans.get_params(deep=True)):
                    out['%s__%s' % (name, key)] = value
            out.update(super(StackEnsembleClassifier, self).get_params(deep=False))
            return out

# Loads a cached model
class PredictionLoader(BaseEstimator):
    def __init__(self, filename):
        self.filename=filename

    def fit(self,X,y=None):
        print("Getting predictions from:",self.filename)
        self.predictions = pickle.load(open(self.filename,'rb'))
        if type(y[0]) is np.int64 or type(y[0]) is np.int32 or type(y[0]) is int: # Because the voter LabelEncodes its 'y's, we have to do it too. We detect that we are bing called by VoterClassifier by checking the type of the label we are to predict. Voters use ints.
            self.predictions = LabelEncoder().fit_transform(self.predictions)
        return self

    def predict(self,X):
        return self.predictions

# Pipeline transformer to turn data from a JSON object into a string - important because Count and TFIDF vectorizers both require strings
class JSONtoString(BaseEstimator,TransformerMixin):
    def __init__(self,remove_spaces=False,use_stemmer=True,remove_symbols=True):
        self.remove_spaces = remove_spaces
        self.use_stemmer = use_stemmer
        self.remove_symbols = remove_symbols
    def fit(self,x,y=None):
        return self
    def transform(self, data):
        if self.remove_symbols:
            rem_sym = lambda s:"".join([ c if c.isalnum() else " " for c in s ])
        else:
            rem_sym = lambda s:s
        if not self.use_stemmer:
            if self.remove_spaces:
                z = [",".join([rem_sym(y).replace(' ','') for y in x['ingredients']]) for x in data]
            else:
                z = [",".join([rem_sym(y) for y in x['ingredients']]).lower() for x in data]
        else:
            from nltk.stem import WordNetLemmatizer
            lemma = WordNetLemmatizer()

            if self.remove_spaces:
                z = [",".join([lemma.lemmatize(rem_sym(y).replace(' ','')) for y in x['ingredients']]) for x in data]
            else:
                z = [",".join([",".join([lemma.lemmatize(yy) for yy in rem_sym(y).split(" ")]) for y in x['ingredients']]).lower() for x in data]
        return z

# TFIDF and Count vectorizers return sparse matrices - this turns them into regular arrays
class DeSparsify(BaseEstimator,TransformerMixin):
#  Anything in the middle of a pipeline needs a fit and a transform,
#  If its only gonna be used in the last step of the pipeline, it can be fit and a predict instead,
#  but you need to inherit from BaseEstimator
    def fit(self,x,y=None):
        return self
    def transform(self, x):
        return x.toarray()

# Wrapper to use DBScan in pipelines
class DBScanTransformer(sklearn.cluster.DBSCAN):
    def transform(self, x):
        return self.fit_predict(x)

# Prints out current state of transformed input in a pipeline
class Printer(BaseEstimator,TransformerMixin):
    def __init__(self,silent=False):
        self.silent=silent
    def fit(self, x, y=None):
        if not self.silent:
            print('Printer.fit() has been called')
        return self
    def transform(self, x):
        if not self.silent:
            print('Printer.transform() has been called')
            try:
                if len(x[0]) < 40:
                    print("Transform step: Samples each look like:",x[0],"\n")
                else:
                    print("Transform step: Samples each look like:",x[0][:39],"...(too long)... \n")
            except:
                print("Transform step: Samples each look like:",x[0],"\n")
        return x


