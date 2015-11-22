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

# Import my stuff from ensembler.py
from ensembler import StackEnsembleClassifier,Printer,DeSparsify,JSONtoBoW


with open('train.json') as f:
    train = json.loads(f.read())

with open('test.json') as f:
    test = json.loads(f.read())

# train = [",".join([y.replace(' ','') for y in x['ingredients']]) for x in data_train]
# test = [",".join([y.replace(' ','') for y in x['ingredients']]) for x in data_test]

train_labels = [x['cuisine'] for x in train]


sub_pipe1 = skpipe.Pipeline([
    ('encoder',skfe.text.CountVectorizer(max_features=100)),
    ('printer3', Printer()),
    ('clf', sklearn.svm.SVC()),
    ])

sub_pipe2 = skpipe.Pipeline([
    ('encoder',skfe.text.CountVectorizer(max_features=1000)),
    ('clf', sklearn.linear_model.LogisticRegression()),
    ])

sub_pipe3 = skpipe.Pipeline([
    ('encoder',skfe.text.CountVectorizer(max_features=5000)),
    ('selector', sklearn.feature_selection.SelectKBest(sklearn.feature_selection.chi2, 500)),
    ('clf', sklearn.linear_model.LogisticRegression()),
    ])

ensemble = StackEnsembleClassifier(
    [
        ('pipe1', sub_pipe1),
        ('pipe2', sub_pipe2),
        ('pipe3', sub_pipe3),

    ],
    ('top', sklearn.linear_model.LogisticRegression())
)

pipeline = skpipe.Pipeline([
    ('printer1', Printer()),
    ('BoW', JSONtoBoW()),
    ('printer2', Printer()),
    ('ensemble', ensemble),
])

def compare_models(pipeline):
    print("Performing comparison:")
    # Lets do a show down between the old model and the new one
    encoder = skfe.text.CountVectorizer(max_features=1000)
    old_scores = sklearn.cross_validation.cross_val_score(sklearn.linear_model.LogisticRegression(), encoder.fit_transform([",".join([y.replace(' ','') for y in x['ingredients']]) for x in train]), train_labels,cv=5,n_jobs=-1)

    # Next two lines doing the same thing as the CV line above,
    # I just wanted to turn off the printing for the comparison and set the CountVectorizer to 1000 for the showdown so its a fair fight
    # GridSearch is handy for temporarily setting parameters in the pipeline, so later we can call pipeline.fit() and still have it print out info and stuff
    # Also I'll throw in an example of setting a nested parameter by setting the n_estimators of the GradientBoostingClassifier in the ensemble
    # 'ensemble__item2__n_estimators':[15],
    clf = sklearn.grid_search.GridSearchCV(pipeline,{},cv=5,n_jobs=-1,verbose=5)
    clf.fit(train,train_labels)

    print("Old best model average score over 5 folds is:" , np.mean(old_scores))
    print("Pipeline average score over 5 folds is:" , np.mean(clf.best_score_))

def fit_and_save_scores(pipeline):

    print("Fitting Pipeline:\n")
    pipeline.fit(train,train_labels)
    print("\nDone Fitting, running predict:\n")
    predictions = pipeline.predict(test)

    exit()

    ids = [x['id'] for x in test]

    final_str = "id,cuisine"
    for idx,i in enumerate(ids):
        final_str+="\n"
        final_str+=str(i)
        final_str+=","
        final_str+=predictions[idx]

    with open('output.csv','w') as f:
        f.write(final_str)


compare_models(pipeline)
# fit_and_save_scores(pipeline)