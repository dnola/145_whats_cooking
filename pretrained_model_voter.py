__author__ = 'davidnola'


import json
import sklearn.pipeline as skpipe
import sklearn.feature_extraction as skfe
import sklearn.linear_model
import sklearn.svm
import sklearn.ensemble
import sklearn.cross_validation
import sklearn.cross_validation
import sklearn.grid_search
import sklearn.feature_selection
import sklearn.neighbors
import sklearn.cluster
import numpy as np
import pickle

from pipeline_helpers import Printer,DeSparsify,JSONtoString,PredictionLoader

# Important Note: When using a pretrained model, the model was trained on all of the training data, meaning there is no way to accurately cross validate
# If you want to cross validate, you will need to actually use the pipeline in deep_net.py - which is very time consuming
# And more or less needs CUDA if you want it to finish in a reasonable amount of time
# So we can't grid search or anything with this script. Our best bet is to either A - do a giant time consuming grid search while not using pretrained models,
# or B - optimize the pipelines separately without the neural net, only adding in the pretrained net to the voter for the final prediction - as we do here


############################################################# Loading data ##############################################################################

# Load up data. We transform it inside the pipeline now, so no need to preprocess
with open('train.json', encoding='utf-8') as f:
    train = json.loads(f.read())

with open('test.json', encoding='utf-8') as f:
    test = json.loads(f.read())

train_labels = [x['cuisine'] for x in train]

####################################### Base pipeline construction ##############################################################################

silent = True # Set this to true to shut up the Printer()s

net = PredictionLoader('net_predictions.pkl')

sub_pipe0 = skpipe.Pipeline([
    ('ingredient_string', JSONtoString(remove_spaces=False)),
    ('printer0', Printer(silent)), # This prints out what the data currently looks like. Pipelines work by sequentially transforming the data step by step - so this Printer() will help you to see how those transformations go
    ('encoder',skfe.text.TfidfVectorizer(strip_accents='unicode',stop_words='english')),
    ('printer1', Printer(silent)),
    ('desparsify', DeSparsify()), # not necessary to desparsify for SVC, but helps you see how pipelines work
    ('printer2', Printer(silent)),
    ('clf', sklearn.svm.LinearSVC()),
    ])


sub_pipe1 = skpipe.Pipeline([
    ('ingredient_string', JSONtoString(remove_spaces=False)),
    ('printer0', Printer(silent)),
    ('encoder',skfe.text.CountVectorizer(strip_accents='unicode',stop_words='english')),
    ('printer1', Printer(silent)),
    ('desparsify', DeSparsify()),
    ('printer2', Printer(silent)),
    ('clf', sklearn.linear_model.LogisticRegression()),
    ])


sub_pipe2 = skpipe.Pipeline([
    ('ingredient_string', JSONtoString(remove_spaces=False)),
    ('printer0', Printer(silent)),
    ('encoder',skfe.text.TfidfVectorizer(strip_accents='unicode',stop_words='english')),
    ('reducer',sklearn.decomposition.TruncatedSVD(n_components=2000)), # This is just one method of dimensionality reduction - its similar to clustering features (even closer to PCA if you know what that is), SVD is known to be good for text data
    ('normalizer',sklearn.preprocessing.Normalizer()), # reduce decomposed features to 0 mean, unit variance
    ('printer1', Printer(silent)),
    ('clf', sklearn.svm.LinearSVC()),
    ])


# Heres an example of using feature union to combine two representations - it basically just lines up the sets of features one next to the other
sub_pipe3 = skpipe.Pipeline([
    ('ingredient_string', JSONtoString(remove_spaces=False)),
    ('union', skpipe.FeatureUnion([
        ('encoder1',skfe.text.CountVectorizer(strip_accents='unicode',stop_words='english',max_features=1500)),
        ('encoder2',skfe.text.TfidfVectorizer(strip_accents='unicode',stop_words='english',max_features=1500)),
        ])),
    ('clf', sklearn.ensemble.ExtraTreesClassifier(n_estimators=300,n_jobs=-1)),
    ])




####################################### Voter pipeline construction ###################################################################


# Now we put them together in an ensemble...
base_layer = [
        ('pipe0', sub_pipe0),
        ('pipe1', sub_pipe1),
        ('pipe2', sub_pipe2),
        ('pipe3', sub_pipe3),
        ('net', net)
    ]

# Most unnecessary pipeline ever...
voter_pipeline = skpipe.Pipeline([
    ('voter', sklearn.ensemble.VotingClassifier(base_layer)),
])
print("Fitting voter...")
voter_pipeline.fit(train,train_labels)
print("Getting predictions...")

predictions = voter_pipeline.predict(test)

ids = [x['id'] for x in test]

final_str = "id,cuisine"
for idx,i in enumerate(ids):
    final_str+="\n"
    final_str+=str(i)
    final_str+=","
    final_str+=predictions[idx]

with open('voter_with_pretrained_net_output.csv','w') as f:
    f.write(final_str)

