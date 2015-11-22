__author__ = 'davidnola'

import json
import xgboost
import sklearn.pipeline as skpipe
import sklearn.preprocessing as skpre
import sklearn.feature_extraction as skfe
from sklearn.base import TransformerMixin, BaseEstimator
import sklearn.svm
import sklearn.linear_model
import sklearn.ensemble
import sklearn.multiclass as skmulti
import numpy as np
import itertools

class Ensembler(TransformerMixin):
    def __init__(self, model_name="LogReg",**kwargs):
        if 'SVM' in model_name:
            self.model = sklearn.svm.LinearSVC(**kwargs)
        if 'LogReg' in model_name:
            self.model = sklearn.linear_model.LogisticRegression(**kwargs)
        if 'Forest' in model_name:
            self.model = sklearn.ensemble.RandomForestClassifier(**kwargs)

    def fit(self, x, y=None):
        x = x.toarray()
        chop = int(9*len(x)/10)
        x=x[:chop]
        self.model.fit(x,y[:chop])
        return self

    def transform(self, x):
        x = x.toarray()
        return [self.model.predict([z]) for z in x]

class StackLabelEncoder(TransformerMixin):
    def __init__(self, stack_encoder,**kwargs):
        self.stack_encoder = stack_encoder

    def fit(self, x, y=None):
        return self.stack_encoder

class Stacker(BaseEstimator):
    def __init__(self, model_name="LogReg",**kwargs):
        self.stack_encoder = stack_encoder
        if 'SVM' in model_name:
            self.model = sklearn.svm.LinearSVC(**kwargs)
        if 'LogReg' in model_name:
            self.model = sklearn.linear_model.LogisticRegression(**kwargs)
        if 'Forest' in model_name:
            self.model = sklearn.ensemble.RandomForestClassifier(**kwargs)

    def fit(self, x, y=None):
        x=x.toarray()
        chop = int(9*len(x)/10)
        x = x[chop:]
        self.model.fit(x,y[chop:])
        return self

    def predict(self, x):
        return self.model.predict(x)

class Printer(TransformerMixin):
    def fit(self, x, y=None):
        #print("Fit: Samples each look like:",x[0])
        # if y!=None:
        #     print("Fit: Labels each look like:",y[0])
        return self
    def transform(self, x):
        print("Transform: Samples each look like:",x[0])
        return x
    def predict(self, x):
        print("Predict: Outputs each look like:",x[0])
        return x


with open('train.json') as f:
    data_train = json.loads(f.read())

with open('test.json') as f:
    data_test = json.loads(f.read())

train = [",".join([y.replace(' ','') for y in x['ingredients']]) for x in data_train]
test = [",".join([y.replace(' ','') for y in x['ingredients']]) for x in data_test]

train_labels = [x['cuisine'] for x in data_train]



label_encoder = skpre.LabelEncoder()
label_one_hot = skpre.OneHotEncoder()
label_one_hot.fit(list(map(lambda x:[x], label_encoder.fit_transform(train_labels))))
stack_encoder = lambda x : label_one_hot.transform(list(map(lambda x:[x],label_encoder.transform(x)))).toarray()

ensemble = skpipe.FeatureUnion([
    ('ens1',Ensembler(model_name='LogReg',C=1)),
    ('ens2',Ensembler(model_name='Forest',n_estimators=20)),
])

pipeline = skpipe.Pipeline([
    ('printer0', Printer()),
    ('encoder',skfe.text.CountVectorizer(max_features=100)),
    ('printer1', Printer()),
    ('ensemble', ensemble),
    ('printer2', Printer()),

])

print("Fitting Pipeline:")
pipeline.fit(train,train_labels)
print("Done Fitting, running predict:")
predictions = pipeline.predict(test)

exit()

ids = [x['id'] for x in data_test]

final_str = "id,cuisine"
for idx,i in enumerate(ids):
    final_str+="\n"
    final_str+=str(i)
    final_str+=","
    final_str+=predictions[idx]

with open('output.csv','w') as f:
    f.write(final_str)
