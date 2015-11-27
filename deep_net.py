__author__ = 'davidnola'

import pickle
import json

import theano
import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
import sklearn.grid_search
from lasagne.init import GlorotUniform
import numpy as np
import sklearn.pipeline as skpipe
import sklearn.feature_extraction as skfe
import sklearn.cross_validation
from lasagne.layers import DropoutLayer


from pipeline_helpers import Printer,DeSparsify,JSONtoString
from deep_net_helpers import float32, EarlyStopping,PipelineNet
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

with open('train.json', encoding='utf-8') as f:
    train = json.loads(f.read())

with open('test.json', encoding='utf-8') as f:
    test = json.loads(f.read())

train_labels = [x['cuisine'] for x in train]


# Neural Nets are literally voodoo. Don't even ask me how I got to the hyperparameters I did, I just kept trying stuff until I could see valid loss decreasing significantly each iteration
# Nobody knows how neural nets really theoretically work. Not even the pros. Not even the guys who pioneered them. Nobody.
net = PipelineNet(
    #layer list:
    [
        ('input', layers.InputLayer),
        ('dense', layers.DenseLayer),
        ('maxout', layers.FeaturePoolLayer),
        ('dropout0', DropoutLayer),
        ('dense2', layers.DenseLayer),
        ('maxout2', layers.FeaturePoolLayer),
        ('dropout1', DropoutLayer),
        ('output', layers.DenseLayer),
        ],

    # layer specs:
    dense_num_units=1024,dense_W=GlorotUniform(),
    maxout_pool_size=2,
    dropout0_p=.2,
    dense2_num_units=1024,dense2_W=GlorotUniform(),
    maxout2_pool_size=2,
    dropout1_p=.2,

    # network hyperparams:

    on_epoch_finished=[EarlyStopping()],
 
    update=nesterov_momentum,
    update_learning_rate=theano.shared(float32(0.01)), # How much to scale current epochs gradient when updating weights - too high and you overshoot minimum
    update_momentum=theano.shared(float32(0.90)), # Use a some of last epoch's gradient as well when updating weights - too high and you overshoot minimum
 
    regression=True, # Always just set this to true. Even when you aren't doing regression.
    max_epochs=200,
    )

##############

# label_encoder = LabelEncoder()
# one_hot = OneHotEncoder()
#
# # train_labels = list(map(lambda x:[x],label_encoder.fit_transform(train_labels)))
# # train_labels = np.array(one_hot.fit_transform(train_labels).toarray(),dtype=np.float32)
# train = JSONtoString().fit_transform(train)
# train = skfe.text.CountVectorizer(strip_accents='unicode',stop_words='english').fit_transform(train).toarray()
# train = np.array(train,dtype=np.float32)
#
# model = sklearn.grid_search.GridSearchCV(net,{},cv=3,verbose=10,n_jobs=1)
# model.fit(train,train_labels)
#
#
# print("Average score over 3 folds is:" , model.best_score_)
#
# exit()

##############
pipe = skpipe.Pipeline([
    ('stringify_json', JSONtoString()),
    ('printer1', Printer()),
    ('encoder',skfe.text.CountVectorizer(strip_accents='unicode',stop_words='english')),
    ('printer2', Printer()),
    ('desparsify', DeSparsify()), # not necessary to desparsify for SVC, but helps you see how pipelines work
    ('printer3', Printer()), # Note that tfidf is sparse, so most values are zero
    ('clf', net),
    ])

model = sklearn.grid_search.GridSearchCV(pipe,{},cv=2,n_jobs=1,verbose=10)
model.fit(train,train_labels)


print("Average score over 2 folds is:" , model.best_score_)
# pipe.fit(train,train_labels)

preds = model.predict(test)

ids = [x['id'] for x in test]

final_str = "id,cuisine"
for idx,i in enumerate(ids):
    final_str+="\n"
    final_str+=str(i)
    final_str+=","
    final_str+=preds[idx]

with open('net_output.csv','w') as f:
    f.write(final_str)