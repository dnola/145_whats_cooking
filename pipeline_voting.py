__author__ = 'davidnola'

__author__ = 'davidnola'

import json
import sklearn.pipeline as skpipe
import sklearn.preprocessing as skpre
import sklearn.feature_extraction as skfe
import sklearn.linear_model
import sklearn.svm
import sklearn.ensemble
import sklearn.cross_validation
import numpy as np
import sklearn.cross_validation
from sklearn.metrics import accuracy_score
import sklearn.grid_search
import sklearn.feature_selection
import sklearn.neighbors
import sklearn.cluster
# Import my stuff from pipeline_helpers.py
from pipeline_helpers import StackEnsembleClassifier,Printer,DeSparsify,JSONtoString,DBScanTransformer


# Load up data. We transform it inside the pipeline now, so no need to preprocess
with open('train.json') as f:
    train = json.loads(f.read())

with open('test.json') as f:
    test = json.loads(f.read())

train_labels = [x['cuisine'] for x in train]


# Build a base layer of classifiers - we check CV for each of them individually to see if it would be better to just use one of the classifiers instead of ensembling them
sub_pipe0 = skpipe.Pipeline([
    ('encoder',skfe.text.TfidfVectorizer(strip_accents='unicode',stop_words='english')),
    ('printer1', Printer()),
    ('desparsify', DeSparsify()),
    ('printer2', Printer()),
    ('clf', sklearn.svm.LinearSVC()),
    ])
# print('sub0',np.mean(sklearn.cross_validation.cross_val_score(sub_pipe0, JSONtoBoW().transform(train), train_labels,cv=5,n_jobs=1)))

sub_pipe1 = skpipe.Pipeline([
    ('encoder',skfe.text.CountVectorizer(strip_accents='unicode',stop_words='english')),
    ('printer1', Printer()),
    ('desparsify', DeSparsify()),
    ('printer2', Printer()),
    ('clf', sklearn.linear_model.LogisticRegression()),
    ])
# print('sub1',np.mean(sklearn.cross_validation.cross_val_score(sub_pipe1, JSONtoBoW().transform(train), train_labels,cv=5,n_jobs=1)))

sub_pipe2 = skpipe.Pipeline([
    ('encoder',skfe.text.TfidfVectorizer(strip_accents='unicode',stop_words='english')),
    ('reducer',sklearn.decomposition.TruncatedSVD(n_components=2000)),
    ('normalizer',sklearn.preprocessing.Normalizer()),
    ('printer', Printer()),
    ('clf', sklearn.svm.LinearSVC()),
    ])
# print('sub2',np.mean(sklearn.cross_validation.cross_val_score(sub_pipe2, JSONtoBoW().transform(train), train_labels,cv=5,n_jobs=1))) # Multiprocessing the SVD seems to break things so were just gonna use n_jobs=1


# Heres an example of using feature union to combine two representations - it basically just lines up the sets of featrues one next to the other
sub_pipe3 = skpipe.Pipeline([
    ('union', skpipe.FeatureUnion([
        ('encoder1',skfe.text.CountVectorizer(strip_accents='unicode',stop_words='english')),
        ('encoder2',skfe.text.TfidfVectorizer(strip_accents='unicode',stop_words='english')),
        ])),
    ('clf', sklearn.ensemble.ExtraTreesClassifier(n_estimators=500,n_jobs=-1)),
    ])
print('sub3',np.mean(sklearn.cross_validation.cross_val_score(sub_pipe3, JSONtoString().transform(train), train_labels,cv=5,n_jobs=1)))


# Now we put them together in an ensemble...
base_layer = [
        ('pipe0', sub_pipe0),
        ('pipe1', sub_pipe1),
        ('pipe2', sub_pipe2),
        ('pipe3', sub_pipe3),
    ]

# And build a pipeline to preprocess the data and pass it into the voter
voter_pipeline = skpipe.Pipeline([
    ('printer1', Printer()),
    ('union', skpipe.FeatureUnion([ # Should we treat "green beans" as a single ingredient "greenbeans", or two seperate ingredients "green" and "beans"?
        ('BoW', JSONtoString(remove_spaces=True)), # Use a feature union because...
        ('BoW', JSONtoString(remove_spaces=False)), # WHY NOT BOTH!
        ])),
    ('printer2', Printer()),
    ('voter', sklearn.ensemble.VotingClassifier(base_layer)),
])


print('Voter with default params scores',np.mean(sklearn.cross_validation.cross_val_score(voter_pipeline, train, train_labels,cv=7,n_jobs=1)))
# At least in my tests, the voter outperforms any single model on its own

