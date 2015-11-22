__author__ = 'davidnola'

from sklearn.base import BaseEstimator
from sklearn.externals.six import iteritems
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

class StackEnsembleClassifier(BaseEstimator):

    def __init__(self, base_classifiers, stack_classifier,hold_out_percent = .9,categorical_labels=True):
        self.base_classifiers = base_classifiers
        self.all_items = stack_classifier+[stack_classifier]
        self.hold_out_percent = hold_out_percent
        self.categorical=categorical_labels


    def fit(self, X, y=None):
        if self.categorical:
            label_encoder = LabelEncoder()
            one_hot = OneHotEncoder()
            label_encoder.fit(y)
            one_hot.fit(label_encoder.transform(y))
            stack_encoder = lambda x: one_hot.transform(list(map(lambda x:[x],label_encoder.transform(x)))).toarray()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-self.hold_out_percent)



    def predict(self, X):
        pass

    def get_params(self, deep=True):
        if not deep:
            return super(StackEnsembleClassifier, self).get_params(deep=False)
        else:
            out = dict(self.all_items)
            for name, trans in self.all_items:
                for key, value in iteritems(trans.get_params(deep=True)):
                    out['%s__%s' % (name, key)] = value
            out.update(super(StackEnsembleClassifier, self).get_params(deep=False))
            return out



# label_encoder = skpre.LabelEncoder()
# label_one_hot = skpre.OneHotEncoder()
# label_one_hot.fit(list(map(lambda x:[x], label_encoder.fit_transform(train_labels))))
# stack_encoder = lambda x : label_one_hot.transform(list(map(lambda x:[x],label_encoder.transform(x)))).toarray()