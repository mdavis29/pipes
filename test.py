def make_classifier():
    import pandas as pd
    import xgboost as xgb
    import numpy as np
    from sklearn import model_selection
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_breast_cancer
    from pipes.pipes import Bagpipe, DMatrixEstimator
    b = load_breast_cancer()
    X = b.data
    Y = np.array(b.target, dtype=np.float)
    feature_names = b.feature_names
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.33, random_state=2012)
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names)
    # fit model no training data
    params = {'max_depth': 2, 'eta': .3, 'silent': 1, 'objective': 'binary:logistic', 'eval_metric': 'auc'}
    model = xgb.train(params, dtrain, evals=[(dtest, 'test')], )
