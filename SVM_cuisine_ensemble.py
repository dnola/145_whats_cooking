# A failed attempt at increasing our score by hyperparameter optimzizing each cuisine instead of coming up with a global set of hyperparameters.
# Took too long to run - so unfortunately not a a part of our final solution or report.


def do_single_fit(args):
    import copy
    (key,search,X,y) = args
    return (key,copy.deepcopy(search).fit(X, [1 if key in lab else 0 for lab in y]))

class BinarizedLabelOptimizer:
    def fit(self,X,y=None):
        self.labels = np.unique(y)

        params = {
            'stringify_json__remove_spaces':[True,False],
            'stringify_json__remove_symbols':[True,False],
            'stringify_json__use_stemmer':[True,False],
            'encoder__ngram_range':[(1,1),(1,2),(1,3),(1,4),(1,5)],
            'encoder__max_df':[1,.90,.85,.80,.70,.60,.50],
            'clf__C':[.5,.6,.7,.8,.9,1],
            'clf__class_weight':['balanced',None,None,None]
        }

        pipe = skpipe.Pipeline([ # This has values generated by randomized_mass_search
            ('stringify_json', JSONtoString(remove_spaces=False)),
            ('encoder',skfe.text.TfidfVectorizer(strip_accents='unicode',stop_words='english',max_df=.7,sublinear_tf=True)),
            # ('desparse', DeSparsify()),
            # ('scaler',sklearn.preprocessing.StandardScaler()),
            ('clf', sklearn.svm.SVC(kernel='linear',probability=True, C=.7))
            # ('clf', sklearn.linear_model.LogisticRegression())
            ])

        search = sklearn.grid_search.RandomizedSearchCV(pipe,params,n_iter=20,n_jobs=1,cv=2,verbose=10)

        self.model_dict_tmp = Parallel(n_jobs=8)(delayed(do_single_fit)((key,search,X,y)) for key in self.labels)

        # self.model_dict_tmp = [(key,copy.deepcopy(search).fit(X, [1 if key in lab else 0 for lab in y])) for key in self.labels]

        self.model_dict = {}
        for k,v in self.model_dict_tmp:
            self.model_dict[k]=v

        for key,value in self.model_dict.items():
            print(key,value.best_score_,value.best_params_)
        return self


    def transform(self,X):
        models = self.model_dict.values()
        toret = np.hstack([np.array(m.predict_proba(X))for m in models])
        return toret

    def predict(self,X): # Based on maximum probability estimate across all cuisine models
        models = self.model_dict.values()
        print(np.array(X).shape)
        preds = np.vstack([np.array([p[1] for p in m.predict_proba(X)]) for m in models]).swapaxes(0,1)
        print(preds[:10])
        print(preds.shape)

        toret = [list(self.model_dict.keys())[x] for x in np.argmax(preds,axis=1)]
        return toret

    def score(self,X,y):
        preds = self.predict(X)
        print(preds[:10])
        print(y[:10])
        return sklearn.metrics.accuracy_score(preds,y)


if __name__ == '__main__':
    import json
    import pickle

    import numpy as np
    import sklearn.cluster
    import sklearn.cross_validation as cv
    import sklearn.cross_validation
    import sklearn.ensemble
    import sklearn.feature_extraction as skfe
    import sklearn.feature_selection
    import sklearn.grid_search
    import sklearn.linear_model
    import sklearn.neighbors
    import sklearn.pipeline as skpipe
    import sklearn.svm
    import sklearn.metrics
    import sklearn.preprocessing
    from joblib import Parallel, delayed

    from sklearn.base import BaseEstimator, TransformerMixin
    import copy
    from pipeline_helpers import JSONtoString, DeSparsify

    with open('train.json') as f:
        train = json.loads(f.read())

    with open('test.json') as f:
        test = json.loads(f.read())

    train_labels = [x['cuisine'] for x in train]

    final_pipe = skpipe.Pipeline([
        ('optimizer', BinarizedLabelOptimizer()),
        # ('ensembler', sklearn.linear_model.LogisticRegression())
    ])

    # X_train, X_test, y_train, y_test = cv.train_test_split(train, train_labels, test_size=0.20,)
    # final_pipe.fit(X_train,y_train)
    # score = final_pipe.score(X_test,y_test)
    # print("Final Score:", score)
    #

    final_pipe.fit(train,train_labels)
    predictions = final_pipe.predict(test)

    pickle.dump(predictions, open('cuisine_ensemble_predictions_short.pkl','wb'))


    ids = [x['id'] for x in test]

    final_str = "id,cuisine"
    for idx,i in enumerate(ids):
        final_str+="\n"
        final_str+=str(i)
        final_str+=","
        final_str+=predictions[idx]

    with open('cuisine_ensemble_output.csv','w') as f:
        f.write(final_str)

    # print("Final Score:", score)
