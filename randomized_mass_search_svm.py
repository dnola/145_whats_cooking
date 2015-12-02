__author__ = 'davidnola'


if __name__ == '__main__':

    import sklearn.pipeline as skpipe
    import sklearn.cross_validation
    import sklearn.grid_search
    import sklearn.feature_extraction as skfe
    import json
    import pickle

    # import my stuff from pipeline_helpers.py
    from pipeline_helpers import JSONtoString
    # Load up data. We transform it inside the pipeline now, so no need to preprocess
    with open('train.json') as f:
        train = json.loads(f.read())

    with open('test.json') as f:
        test = json.loads(f.read())

    train_labels = [x['cuisine'] for x in train]


    pipe0 = skpipe.Pipeline([
        ('stringify_json', JSONtoString()),
        ('encoder',skfe.text.TfidfVectorizer(strip_accents='unicode',stop_words='english')),
        ('clf', sklearn.svm.LinearSVC()),
        ])

    params0 = {
        'stringify_json__remove_spaces':[True,False],
        'encoder__ngram_range':[(1,1),(1,1),(1,2)],
        'encoder__max_df':[1.0,1.0,.95,.90,.85,.80,.70,.60,.50],
        'encoder__min_df':[1,1,2,3,.1,.2,.05],
        'encoder__max_features':[None,None,3000,2000,1000],
        'encoder__norm' : ['l1','l2'],
        'encoder__sublinear_tf' : [True, False],
        'clf__C':[.1,.3,.5,.6,.7,.8,.9,1,1.5,2,5,100],
        'clf__dual':[True,False],
        'clf__class_weight':['balanced',None,None,None]
    }
    param_scores = []
    for i in range(500):
        grid0 = sklearn.grid_search.RandomizedSearchCV(pipe0,params0,verbose=1,n_jobs=-1,cv=5, n_iter=50,refit=False,error_score=0)
        grid0.fit(train,train_labels)
        best_score = grid0.best_score_
        best_params = grid0.best_params_
        param_scores.append( (best_score,best_params) )

        print("Current Params List:")
        for p in param_scores:
            print(p)
        pickle.dump(param_scores,open('param_scores.pkl','wb'))

