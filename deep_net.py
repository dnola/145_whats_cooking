__author__ = 'davidnola'

# This will only work if you have an Nvidia GPU with CUDA
# You gotta install cuda first, then set an environment variable to the following before running this - (assuming a default cuda install)
# You can set environment variables in PyCharm by going to MenuBar -> Run -> Edit Configurations -> Then click the ... next to environment variables
# Add one names THEANO_FLAGS
# with the value 'nvcc.fastmath=True,floatX=float32,device=gpu,cuda.root=/usr/local/cuda'
# If you aren't using pycharm, just make sure you have the following environment variable set before running the .py
# THEANO_FLAGS = nvcc.fastmath=True,floatX=float32,device=gpu,cuda.root=/usr/local/cuda



import json
import pickle

import sklearn.cross_validation
import sklearn.feature_extraction as skfe
import sklearn.grid_search
import sklearn.pipeline as skpipe
import theano
from lasagne import layers
from lasagne.init import GlorotUniform
from lasagne.layers import DropoutLayer
import lasagne.nonlinearities
from lasagne.updates import nesterov_momentum

from deep_net_helpers import float32, EarlyStopping,PipelineNet, AdjustVariable
from pipeline_helpers import Printer,DeSparsify,JSONtoString

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
    dense_num_units=10000,dense_W=GlorotUniform(),#dense_nonlinearity=lasagne.nonlinearities.LeakyRectify(),
    maxout_pool_size=2,
    dropout0_p=.3,

    dense2_num_units=500,dense2_W=GlorotUniform(),#dense2_nonlinearity=lasagne.nonlinearities.LeakyRectify(),
    maxout2_pool_size=2,
    dropout1_p=.1,



    # network hyperparams:

    on_epoch_finished=[EarlyStopping(),AdjustVariable('update_learning_rate', start=0.05, stop=0.0001),AdjustVariable('update_momentum', start=0.9, stop=0.99)],
 
    update=nesterov_momentum,
    update_learning_rate=theano.shared(float32(0.1)), # How much to scale current epochs gradient when updating weights - too high and you overshoot minimum
    update_momentum=theano.shared(float32(0.90)), # Use a some of last epoch's gradient as well when updating weights - too high and you overshoot minimum
 
    regression=True, # Always just set this to true. Even when you aren't doing regression.
    max_epochs=750,
    verbose=1
    )


# Preprocessing pipeline
pipe = skpipe.Pipeline([
    ('stringify_json', JSONtoString(remove_spaces=False,use_stemmer=True,remove_symbols=True)),
    ('printer1', Printer()),
    ('encoder',skfe.text.TfidfVectorizer(strip_accents='unicode',stop_words='english')),
    ('printer2', Printer()),
    ('desparsify', DeSparsify()), # not necessary to desparsify for SVC, but helps you see how pipelines work
    ('printer3', Printer()), # Note that tfidf is sparse, so most values are zero
    ('clf', net),
    ])

model = sklearn.grid_search.RandomizedSearchCV(pipe,{'clf__dense_num_units':[1000,2000,500],
                                                     'clf__dense2_num_units':[500,200,1000],'encoder__max_df':[1.0,.8, .6],
                                                     'clf__dropout0_p':[.3,.1, 0],'clf__maxout_pool_size':[2,4],'clf__dropout1_p':[.3,.1, 0],'clf__maxout2_pool_size':[2,4],'stringify_json__use_stemmer':[True,False]},cv=2,n_jobs=1,verbose=10, n_iter=18)
model.fit(train,train_labels)

print("Average score over 2 folds is:" , model.best_score_)


# Now generate predictions and dump them to disk, as well as submittable CSV form

preds = model.predict(test)

pickle.dump(preds,open('net_predictions3.pkl','wb'))

print("Best params::" , model.best_params_)

print("writing to disk...")
ids = [x['id'] for x in test]

final_str = "id,cuisine"
for idx,i in enumerate(ids):
    final_str+="\n"
    final_str+=str(i)
    final_str+=","
    final_str+=preds[idx]

with open('net_output.csv','w') as f:
    f.write(final_str)

print("Best params::" , model.best_params_)
print("Best score over 2 folds is:" , model.best_score_)