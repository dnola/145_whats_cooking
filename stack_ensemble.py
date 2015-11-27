__author__ = 'davidnola'

"""

Currently the voter outperforms any configuration of a stack ensemble I can come up with
Which is sad because stack ensembles are cool when they work
But they only work like half the time...

"""


import json
import sklearn.pipeline as skpipe
import sklearn.feature_extraction as skfe
import sklearn.linear_model
import sklearn.svm
import sklearn.ensemble
import sklearn.cross_validation
import numpy as np
import sklearn.cross_validation
import sklearn.grid_search
import sklearn.feature_selection
import sklearn.neighbors
import sklearn.cluster
# Import my stuff from pipeline_helpers.py
from pipeline_helpers import StackEnsembleClassifier, JSONtoString


# Load up data. We transform it inside the pipeline now, so no need to preprocess
with open('train.json') as f:
    train = json.loads(f.read())

with open('test.json') as f:
    test = json.loads(f.read())

train_labels = [x['cuisine'] for x in train]


# Build a base layer of classifiers - we check each of them individually to see if it would be better to just use one of the classifiers instead of ensembling them
sub_pipe0 = skpipe.Pipeline([
    ('encoder',skfe.text.TfidfVectorizer(strip_accents='unicode',stop_words='english')),
    # ('printer', Printer()),
    # ('desparsify', DeSparsify()),
    # ('printer', Printer()),
    ('clf', sklearn.svm.LinearSVC()),
    ])
print("Score to beat:")
print('sub0',np.mean(sklearn.cross_validation.cross_val_score(sub_pipe0, JSONtoString().transform(train), train_labels,cv=7,n_jobs=-1)))

sub_pipe1 = skpipe.Pipeline([
    ('encoder',skfe.text.CountVectorizer(strip_accents='unicode',stop_words='english')),
    # ('printer', Printer()),
    ('clf', sklearn.linear_model.LogisticRegression()),
    ])
# print('sub1',np.mean(sklearn.cross_validation.cross_val_score(sub_pipe1, JSONtoBoW().transform(train), train_labels,cv=7,n_jobs=-1)))

sub_pipe2 = skpipe.Pipeline([
    ('encoder',skfe.text.TfidfVectorizer(strip_accents='unicode',stop_words='english')),
    ('reducer',sklearn.decomposition.TruncatedSVD(n_components=2000)),
    ('normalizer',sklearn.preprocessing.Normalizer()),
    # ('printer', Printer()),
    ('clf', sklearn.svm.LinearSVC()),
    ])
# print('sub2',np.mean(sklearn.cross_validation.cross_val_score(sub_pipe2, JSONtoBoW().transform(train), train_labels,cv=5,n_jobs=1)))


sub_pipe3 = skpipe.Pipeline([
    ('encoder',skfe.text.CountVectorizer(strip_accents='unicode',stop_words='english')),
    ('clf', sklearn.ensemble.ExtraTreesClassifier(n_estimators=500,n_jobs=-1)),
    ])
# print('sub3',np.mean(sklearn.cross_validation.cross_val_score(sub_pipe3, JSONtoBoW().transform(train), train_labels,cv=7,n_jobs=1)))

sub_pipe4 = skpipe.Pipeline([
    ('encoder',skfe.text.TfidfVectorizer(strip_accents='unicode',stop_words='english')),
    ('reducer',sklearn.decomposition.TruncatedSVD(n_components=100)),
    ('clf', sklearn.ensemble.GradientBoostingClassifier(n_estimators=50, learning_rate=1.0,max_depth=1)),
    ])
# print('sub4',np.mean(sklearn.cross_validation.cross_val_score(sub_pipe4, JSONtoBoW().transform(train), train_labels,cv=3,n_jobs=1)))


base_layer =     [
        ('pipe0', sub_pipe0),
        ('pipe1', sub_pipe1),
        # ('pipe2', sub_pipe2),
        ('pipe3', sub_pipe3),
        # ('pipe4', sub_pipe4),
    ]

# A stack ensemble uses the outputs of a base layer of classifiers as the features for a top layer classifier
# So in the logistic regression case, instead of just having all the models vote,
# we try to learn a weight to assign to each model
# The problem is we need to reserve some data just for the top layer to learn these weights
# So our bottom layer predictors become weaker because they can't use that data - which is why sometimes voting beats stack ensemble
# Because voting doesn't have this requirement - the base classifiers can use all data
# Stack ensembles tend to win when you can get base level models that are unique enough - the models I have above are
# all just too similar
ensemble = StackEnsembleClassifier(
    base_layer,
    ('top', sklearn.feature_selection.RFECV(sklearn.linear_model.LogisticRegression(penalty='l1')))
    # Recursive Feature Elimination - it keeps trying to get rid of features 1 by 1 to see if it improves CV score
    # The theory here is that in the stack, some base classifiers will be bad at predicting certain labels
    # So we get rid of those predictions
)

pipeline = skpipe.Pipeline([
    # ('printer1', Printer()),
    ('BoW', JSONtoString()),
    # ('printer2', Printer()),
    ('ensemble', ensemble),
])

def compare_models(pipeline):
    print("Performing comparison:")
    # Lets do a show down between the old model we first submitted, and the new one
    encoder = skfe.text.CountVectorizer(max_features=1000)
    old_scores = sklearn.cross_validation.cross_val_score(sklearn.linear_model.LogisticRegression(), encoder.fit_transform([",".join([y.replace(' ','') for y in x['ingredients']]) for x in train]), train_labels,cv=5,n_jobs=-1)

    params = [{
        'ensemble__top__estimator__penalty' : ['l1', 'l2'],
        'ensemble__top__estimator__C' : [.1,.3,.5,1],
        'ensemble__refit_base': [True,False],
        'ensemble__hold_out_percent':[.9,.8,.7]
    }]

    # params = {} # use default params

    clf = sklearn.grid_search.GridSearchCV(pipeline,params,cv=3,n_jobs=1,verbose=5)
    clf.fit(train,train_labels)

    print("Best params:",clf.best_params_)

    print("Original best model average score over 5 folds is:" , np.mean(old_scores))
    print("Pipeline average score over 5 folds is:" , np.mean(clf.best_score_))

def fit_and_save_scores(pipeline):

    print("Fitting Pipeline:\n")
    pipeline.fit(train,train_labels)
    print("\nDone Fitting, running predict:\n")
    predictions = pipeline.predict(test)


    ids = [x['id'] for x in test]

    final_str = "id,cuisine"
    for idx,i in enumerate(ids):
        final_str+="\n"
        final_str+=str(i)
        final_str+=","
        final_str+=predictions[idx]

    with open('ensemble_output.csv','w') as f:
        f.write(final_str)


compare_models(pipeline)
fit_and_save_scores(pipeline)