

import xgboost as xgb
import sklearn.pipeline as skpipe
import sklearn.cross_validation
import sklearn.feature_extraction as skfe
import numpy as np
import json

from pipeline_helpers import Printer,DeSparsify,JSONtoString
from xgboost_helpers import XGBClassifier

with open('train.json') as f:
    train = json.loads(f.read())

with open('test.json') as f:
    test = json.loads(f.read())

train_labels = [x['cuisine'] for x in train]


pipe = skpipe.Pipeline([
    ('stringify_json', JSONtoString()),
    ('encoder',skfe.text.TfidfVectorizer(strip_accents='unicode',stop_words='english')),
    ('desparsify', DeSparsify()),
    ('clf', xgb.XGBClassifier(max_depth=20,objective="multi:softmax",n_estimators=10,silent=False)),
    ])

print("Fitting pipeline:")
print('pipe score:',np.mean(sklearn.cross_validation.cross_val_score(pipe, train, train_labels,cv=2,n_jobs=-1,verbose=10))) #CV=2, so you will see each pipeline run 2 times
