__author__ = 'davidnola'

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
from pipeline_helpers import StackEnsembleClassifier,Printer,DeSparsify,JSONtoString,DBScanTransformer

############################################################# Loading data ##############################################################################

# Load up data. We transform it inside the pipeline now, so no need to preprocess
with open('train.json') as f:
    train = json.loads(f.read())

with open('test.json') as f:
    test = json.loads(f.read())

train_labels = [x['cuisine'] for x in train]

####################################### Base pipeline construction and testing ##############################################################################

silent = False # Set this to true to shut up the Printer()s


# Build a base layer of classifiers - we check CV for each of them individually to see if it would be better to just use one of the classifiers instead of ensembling them

print('First, check pipeline by pipeline to make sure voter ensembling them all actually does better than just using one')
print('\ntesting pipe0...')
sub_pipe0 = skpipe.Pipeline([
    ('printer0', Printer(silent)), # This prints out what the data currently looks like. Pipelines work by sequentially transforming the data step by step - so this Printer() will help you to see how those transformations go
    ('encoder',skfe.text.TfidfVectorizer(strip_accents='unicode',stop_words='english')),
    ('printer1', Printer(silent)),
    ('desparsify', DeSparsify()), # not necessary to desparsify for SVC, but helps you see how pipelines work
    ('printer2', Printer(silent)),
    ('clf', sklearn.svm.LinearSVC()),
    ])
print('pipe0 score:',np.mean(sklearn.cross_validation.cross_val_score(sub_pipe0, JSONtoString().transform(train), train_labels,cv=2,n_jobs=1))) #CV=2, so you will see each pipeline run 2 times

print('\ntesting pipe1...')
sub_pipe1 = skpipe.Pipeline([
    ('printer0', Printer(silent)),
    ('encoder',skfe.text.CountVectorizer(strip_accents='unicode',stop_words='english')),
    ('printer1', Printer(silent)),
    ('desparsify', DeSparsify()),
    ('printer2', Printer(silent)),
    ('clf', sklearn.linear_model.LogisticRegression()),
    ])
print('pipe1 score:',np.mean(sklearn.cross_validation.cross_val_score(sub_pipe1, JSONtoString().transform(train), train_labels,cv=2,n_jobs=1)))

print('\ntesting pipe2...')
sub_pipe2 = skpipe.Pipeline([
    ('printer0', Printer(silent)),
    ('encoder',skfe.text.TfidfVectorizer(strip_accents='unicode',stop_words='english')),
    ('reducer',sklearn.decomposition.TruncatedSVD(n_components=2000)), # This is just one method of dimensionality reduction - its similar to clustering features (even closer to PCA if you know what that is), SVD is known to be good for text data
    ('normalizer',sklearn.preprocessing.Normalizer()), # reduce decomposed features to 0 mean, unit variance
    ('printer1', Printer(silent)),
    ('clf', sklearn.svm.LinearSVC()),
    ])
print('pipe2 score:',np.mean(sklearn.cross_validation.cross_val_score(sub_pipe2, JSONtoString().transform(train), train_labels,cv=2,n_jobs=1))) # Multiprocessing the SVD seems to break things so were just gonna use n_jobs=1


print('\ntesting pipe3...')
# Heres an example of using feature union to combine two representations - it basically just lines up the sets of features one next to the other
sub_pipe3 = skpipe.Pipeline([
    ('union', skpipe.FeatureUnion([
        ('encoder1',skfe.text.CountVectorizer(strip_accents='unicode',stop_words='english',max_features=1500)),
        ('encoder2',skfe.text.TfidfVectorizer(strip_accents='unicode',stop_words='english',max_features=1500)),
        ])),
    ('clf', sklearn.ensemble.ExtraTreesClassifier(n_estimators=300,n_jobs=-1)),
    ])
print('pipe3 score:',np.mean(sklearn.cross_validation.cross_val_score(sub_pipe3, JSONtoString().transform(train), train_labels,cv=2,n_jobs=1)))


####################################### Voter pipeline construction and testing ###################################################################

print("\n\nNow lets build our VotingClassifier() and test it\n")
# Now we put them together in an ensemble...
base_layer = [
        ('pipe0', sub_pipe0),
        ('pipe1', sub_pipe1),
        ('pipe2', sub_pipe2),
        ('pipe3', sub_pipe3),
    ]


silent = False # Set this to true to shut up the Printer()s - although I would keep them on here because its the best single step example of how pipelines work

# And build a pipeline to preprocess the data and pass it into the voter
voter_pipeline = skpipe.Pipeline([
    ('printer1', Printer(silent)),
    ('ingredient_string', JSONtoString(remove_spaces=False)),
    ('printer2', Printer(silent)),
    ('voter', sklearn.ensemble.VotingClassifier(base_layer)),
])

# Lets see how the voter model does using the parameters we set above
print('\nVoter with default params scores',np.mean(sklearn.cross_validation.cross_val_score(voter_pipeline, train, train_labels,cv=5,n_jobs=1)))
# At least in my tests, the voter outperforms any single model on its own


####################################### Voter pipeline grid search ##############################################################################

# Now lets run a grid search across a couple parameters, just to see how it works:
# This will probably take a really long time
params = [
    {'ingredient_string__remove_spaces':[True,False]}, # See if default model works best with spaces or without
    {
        'voter__pipe0__encoder__max_features':[3000,5000], # See if limiting the max_features of encoders helps
        'voter__pipe1__encoder__max_features':[3000,5000],
        'voter__pipe2__encoder__max_features':[3000,5000],
    }
]

# If you don't want to wait forever:
# uncomment the following line if you don't want to run a full grid search, and just want to use the parameters specified above
# when the params list is empty, grid search just uses the default parameters

# params = {}

grid_model = sklearn.grid_search.GridSearchCV(voter_pipeline,params,cv=3,n_jobs=1,verbose=5)
grid_model.fit(train,train_labels) # Go take a nap if params isn't {}
print("Best params:",grid_model.best_params_)
print("Grid search best score:" , np.mean(grid_model.best_score_))



####################################### Generate a submission from grid search predictions ########################################################
# Now get our predictions and write them to a file

predictions = grid_model.predict(test)

ids = [x['id'] for x in test]

final_str = "id,cuisine"
for idx,i in enumerate(ids):
    final_str+="\n"
    final_str+=str(i)
    final_str+=","
    final_str+=predictions[idx]

with open('voter_output.csv','w') as f:
    f.write(final_str)