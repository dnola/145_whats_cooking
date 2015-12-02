import json

import numpy as np
import sklearn.cross_validation
import sklearn.feature_extraction as skfe
import sklearn.pipeline as skpipe
import xgboost as xgb
import pickle

from pipeline_helpers import DeSparsify,JSONtoString

with open('train.json', encoding='utf-8') as f:
    train = json.loads(f.read())

with open('test.json', encoding='utf-8') as f:
    test = json.loads(f.read())

train_labels = [x['cuisine'] for x in train]


pipexg = skpipe.Pipeline([
    ('stringify_json', JSONtoString()),
    ('encoder',skfe.text.TfidfVectorizer(strip_accents='unicode',stop_words='english')),
    ('desparsify', DeSparsify()),
    ('clf', xgb.XGBClassifier(max_depth=26,objective="multi:softmax",n_estimators=500,silent=False)),
    ])

print("Fitting pipeline:")
# print('pipe score:',np.mean(sklearn.cross_validation.cross_val_score(pipe, train, train_labels,cv=2,n_jobs=-1,verbose=10))) #CV=2, so you will see each pipeline run 2 times
pipexg.fit(train,train_labels)
pickle.dump(pipexg.predict(test),open('pipexg_predictions.pkl','wb'))