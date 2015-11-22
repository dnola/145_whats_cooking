__author__ = 'davidnola'

import os
import json
import sklearn as sk
import sklearn.linear_model
import sklearn.ensemble
import sklearn.multiclass
import sklearn.preprocessing as skpp
import sklearn.feature_extraction as skfe
import numpy as np


print('Start!')

# Load JSON Data into dicts:
with open('train.json') as f:
    data_train = json.loads(f.read())

with open('test.json') as f:
    data_test = json.loads(f.read())


# Go from JSON to bag of words in 6 lines of code because python is awesome
train = [",".join([y.replace(' ','') for y in x['ingredients']]) for x in data_train]
test = [",".join([y.replace(' ','') for y in x['ingredients']]) for x in data_test]
encoder = skfe.text.CountVectorizer(max_features=1000)

encoder.fit(train+test)
train=list(map(lambda x: x.toarray()[0],encoder.transform(train)))
test=list(map(lambda x: x.toarray()[0],encoder.transform(test)))

# In case we want to one hot encode labels: (sklearn won't make you do it but it could come in handy if you guys want to try other stuff)
# label_encoder = skfe.text.CountVectorizer()
# train_labels = list(map(lambda x: x.toarray()[0], label_encoder.fit_transform([x['cuisine'] for x in data_train])))

# get our training labels:
train_labels = [x['cuisine'] for x in data_train]

########################## DONE WITH PREPROCESSING - LETS BUILD SOME MODELS! ###################################

# A quick tutorial of basic scikit stuff:
def simple_tutorial():

    # First a simple example of how sklearn works with no frills and none of the fancy stuff that makes sklearn unique
    model = sklearn.ensemble.RandomForestClassifier() # How to define an sklearn model
    model.fit(train,train_labels) # How to fit a simple model, no fancy grid search or CV
    preds = model.predict(train[0:5]) # How to get some predictions
    print(train_labels[0:5],preds) # compare to see how we did


    # Now, getting a little fancier lets run a K-fold cross validation and see how we score!
    # Note that LogisticRegression doesn't handle multiple target classes on its own, i.e. it can only predict yes vs no
    # So we put it inside a OnevsRest classifier - which basically lets it pick out one label from a bunch
    model = sk.multiclass.OneVsRestClassifier(sk.linear_model.LogisticRegression())
    scores = sk.cross_validation.cross_val_score(model, train, train_labels,cv=5,verbose=5,n_jobs=-1)
    print("Average score over 5 folds is:" , np.mean(scores))

    # We can try different models and see what does best:

    model = sklearn.ensemble.RandomForestClassifier(n_estimators=100, class_weight='balanced') # Note that we can specify hyperparameters to the tree here, like the number of decision trees in the forest
    # Note in the hyperparameters that we also balance the class weights so its not biased towards predicting the most common class all the time. Sometimes this helps, sometimes it doesn't
    scores = sk.cross_validation.cross_val_score(model, train, train_labels,cv=5,verbose=5,n_jobs=-1)
    print("Average score over 5 folds is:" , np.mean(scores))

simple_tutorial() # Comment this out to skip tutorial


# Okay now lets get fancy! This could take a while computationally...
# We are going to do a 'grid search' - which basically means check a bunch of hyperparameters and see which ones give us the best CV score!

grid_pairs = [
    (sk.multiclass.OneVsRestClassifier(sk.linear_model.LogisticRegression()),[
        {'estimator__C':[.01,.1,.5,1]}, # This is an oddity of scikit - we have to tell it we want to set hyperparameter C of the logistic regression, not of the OVR, so we tell it estimator__C instead of just C to specify that we want to set the estimators hyperparameter instead of OVR
        {'estimator__C':[.01,.1,.5,1], 'estimator__class_weight':['balanced']}, # Note all hyperparameter sets must be lists - even if its only one item
    ]),

    (sk.ensemble.RandomForestClassifier(),[
        {'n_estimators':[20,40,60,80]},
        {'n_estimators':[15,30], 'min_samples_leaf':[1,2,3]},
        {'n_estimators':[15,30], 'class_weight':['balanced_subsample']},
    ]),
]

best_scores = []
best_models = []
for g in grid_pairs:
    model = g[0]
    params = g[1]

    grid_model = sk.grid_search.GridSearchCV(model,params,verbose=5,n_jobs=-1,cv=2) #n_jobs = -1 means split up task and run it on all cores
    grid_model.fit(train, train_labels)

    best_scores.append(grid_model.best_score_)
    best_models.append(grid_model)
    print("Our best score here:", grid_model.best_score_)
    print("Our best params here:",grid_model.best_params_)


######################### OKAY DONE TRAINING - LETS MAKE A SUBMISSION! ###########################


final_model = best_models[np.argmax(best_scores)]

predictions = final_model.predict(test)
ids = [x['id'] for x in data_test]

final_str = "id,cuisine"
for idx,i in enumerate(ids):
    final_str+="\n"
    final_str+=str(i)
    final_str+=","
    final_str+=predictions[idx]

with open('output.csv','w') as f:
    f.write(final_str)


########### POSTSCRIPT AND FINAL NOTES: ###################

# scikit also does a bunch of other kickass stuff I didn't cover here, most importantly Pipelines
# sklearn pipelines are dope. Look them up sometime. They basically let you stack a bunch of models, feature extraction, data augmentation,
# etc together and essentially make your own 'metamodels' comprised of multiple scikit models, encoders, and transformers
# One example of how pipelines could be useful here is that we stack the Vectorizer with another model, so we can use grid search to try and find the best number of max_features to use
# Lots of things to try! Another option is RecursiveFeatureElimination, which scikit does really easy. You basically keep cutting out features and see if certain features are dragging your score down - then you keep only the best features in the end!

