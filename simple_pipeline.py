__author__ = 'davidnola'

import json

import numpy as np
import sklearn.cross_validation
import sklearn.feature_extraction as skfe
import sklearn.pipeline as skpipe


# import my stuff from pipeline_helpers.py
from pipeline_helpers import Printer,DeSparsify,JSONtoString

# Load up data. We transform it inside the pipeline now, so no need to preprocess
with open('train.json') as f:
    train = json.loads(f.read())

with open('test.json') as f:
    test = json.loads(f.read())

train_labels = [x['cuisine'] for x in train]


silent = False # set to True to shut up the printers

pipe = skpipe.Pipeline([
    ('printer0', Printer(silent)), # This prints out what the data currently looks like. Pipelines work by sequentially transforming the data step by step - so this Printer() will help you to see how those transformations go
    ('stringify_json', JSONtoString(remove_spaces=False,use_stemmer=True)),
    ('printer1', Printer(silent)),
    ('encoder',skfe.text.TfidfVectorizer(strip_accents='unicode',stop_words='english')),
    ('printer2', Printer(silent)),
    ('desparsify', DeSparsify()), # not necessary to desparsify for SVC, but helps you see how pipelines work
    ('printer3', Printer(silent)), # Note that tfidf is sparse, so most values are zero
    ('clf', sklearn.svm.LinearSVC()),
    ])

# when .fit() is called on the pipeline, each step of the pipeline has .fit() and .transform() called. When .predict() is called on the pipeline, each step only has .transform() called, because each transformer has already been fittted during .fit()
print("Fitting pipeline:")
pipe.fit(train,train_labels)

input("Press enter to continue on to run predict...")

predictions = pipe.predict(test)

input("Press enter to continue on to check pipeline CV score...")
# Lets see how it does in cross validation:
print('pipe score:',np.mean(sklearn.cross_validation.cross_val_score(pipe, train, train_labels,cv=2,n_jobs=1))) #CV=2, so you will see each pipeline run 2 times

# now write our predicitons to file:



ids = [x['id'] for x in test]

final_str = "id,cuisine"
for idx,i in enumerate(ids):
    final_str+="\n"
    final_str+=str(i)
    final_str+=","
    final_str+=predictions[idx]

with open('pipe_output.csv','w') as f:
    f.write(final_str)