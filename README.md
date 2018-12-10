## Pipes
A package to manage models, pre processing transformations and feature attribution methods


#### Examples:
```python
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from pipes import Bagpipe, DMatrixEstimator


b = load_breast_cancer()
X = b.data
y = np.array(b.target, dtype=np.float)
feature_names = list(b.feature_names)
pipe = Pipeline([('imputer', SimpleImputer(strategy='median'))])
pipe.fit(X)
features = pipe.transform(X)
X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=.33, random_state=2012)
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names)
# fit model no training data
params = {'max_depth': 2, 'eta': .3, 'silent': 1, 'objective': 'binary:logistic', 'eval_metric': 'auc'}
model = xgb.train(params, dtrain, evals=[(dtest, 'test')], )
preds = model.predict(dtrain)

bp = Bagpipe(model = model, pipe=pipe, data=X)
p = bp.predict(X[0:10, :])
bp._transform_deciles(p)
text = bp.feature_contribution(X)
print(text[1])


```