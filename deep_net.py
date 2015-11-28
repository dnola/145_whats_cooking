__author__ = 'davidnola'

# This will only work if you have an Nvidia GPU with CUDA
# You gotta install cuda first, then set an environment variable to the following before running this - (assuming a default cuda install)
# You can set environment variables in PyCharm by going to MenuBar -> Run -> Edit Configurations -> Then click the ... next to environment variables
# Add one names THEANO_FLAGS
# with the value 'nvcc.fastmath=True,floatX=float32,device=gpu,cuda.root=/usr/local/cuda'
# If you aren't using pycharm, just make sure you have the following environment variable set before running the .py
# THEANO_FLAGS = nvcc.fastmath=True,floatX=float32,device=gpu,cuda.root=/usr/local/cuda



import pickle
import json

import theano
from lasagne import layers
from lasagne.updates import nesterov_momentum
import sklearn.grid_search
from lasagne.init import GlorotUniform
import sklearn.pipeline as skpipe
import sklearn.feature_extraction as skfe
import sklearn.cross_validation
from lasagne.layers import DropoutLayer


from pipeline_helpers import Printer,DeSparsify,JSONtoString
from deep_net_helpers import float32, EarlyStopping,PipelineNet

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

        ('output', layers.DenseLayer),
        ],

    # layer specs:
    dense_num_units=2048,dense_W=GlorotUniform(),
    maxout_pool_size=2,
    dropout0_p=.1,

    dense2_num_units=1024,dense2_W=GlorotUniform(),
    maxout2_pool_size=2,

    # network hyperparams:

    on_epoch_finished=[EarlyStopping()],
 
    update=nesterov_momentum,
    update_learning_rate=theano.shared(float32(0.1)), # How much to scale current epochs gradient when updating weights - too high and you overshoot minimum
    update_momentum=theano.shared(float32(0.90)), # Use a some of last epoch's gradient as well when updating weights - too high and you overshoot minimum
 
    regression=True, # Always just set this to true. Even when you aren't doing regression.
    max_epochs=750,
    verbose=1
    )


# Preprocessing pipeline
pipe = skpipe.Pipeline([
    ('stringify_json', JSONtoString()),
    ('printer1', Printer()),
    ('encoder',skfe.text.TfidfVectorizer(strip_accents='unicode',stop_words='english')),
    ('printer2', Printer()),
    ('desparsify', DeSparsify()), # not necessary to desparsify for SVC, but helps you see how pipelines work
    ('printer3', Printer()), # Note that tfidf is sparse, so most values are zero
    ('clf', net),
    ])

model = sklearn.grid_search.GridSearchCV(pipe,{},cv=2,n_jobs=1,verbose=10)
model.fit(train,train_labels)

print("Average score over 2 folds is:" , model.best_score_)

# Now generate predictions and dump them to disk, as well as submittable CSV form

preds = model.predict(test)

pickle.dump(preds,open('net_predictions.pkl','wb'))

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