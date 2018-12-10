from sklearn.base import ClassifierMixin, BaseEstimator


class DMatrixEstimator(ClassifierMixin, BaseEstimator):
    def __init__(self, feature_names):
        self.feature_names = None
        self.n_cols = None

    def fit(self, X,  y=None):
        self.n_cols = X.shape[1]
        return self

    def transform(self, X, y=None):
        from xgboost import DMatrix
        return DMatrix(X, label=y, feature_names=self.feature_names)

class Bagpipe:
    def __init__(self, model, pipe, data=None, features=None, feature_names=None, name=None, verbose=False, **kwargs):
        from datetime import datetime
        import numpy as np
        self.model = model
        self.pipe = pipe
        self.deciles = None
        self.features = features
        self.data = data
        self.feature_names = feature_names
        self.input_names = None
        self.steps = None
        self.data = None
        self.features = None
        self.preds = None
        self.pred_mean = None
        if name is not None:
            self.name = None
        else:
            self.name = 'model :' + str(datetime.now())
        if feature_names is None:
            if verbose:
                print('setting feature names ... ')
            if hasattr(self.model, 'feature_names'):
                self.feature_names = self.model.feature_names
        self.setup(data=data, features=features, verbose=verbose)

    def setup(self, data, features, verbose=False):
        import numpy as np
        if verbose:
            print('transforming features ... ')
        self.transform_features(data, features, verbose=verbose)
        if verbose:
            print('predicting features  ... ')
        self.preds = self.predict(features=self.features)
        self.pred_mean = np.mean(self.preds)
        if verbose:
            print('fitting deciles ... ')
        self._fit_deciles(self.preds)
        if verbose:
            print('completed')

    def _transform_feature_gen(self, data):
        from scipy.sparse import csr_matrix
        if str(type(data)) == "<class 'pandas.core.frame.DataFrame'>":
            for i in range(data.shape[0]):
                temp_data = data.iloc[[i]]
                output = self.pipe.transform(temp_data)
                yield output
        else:
            data_sparse = csr_matrix(data)
            for i in range(data.shape[0]):
                temp_data = data_sparse[i]
                output = self.pipe.transform(temp_data)
                yield output

    def transform_features(self, data=None, features=None, verbose=False):
        from scipy.sparse import vstack
        if data is not None:
            if verbose:
                print('transforming ...')
            features_list = list(self._transform_feature_gen(data))
            output = vstack(features_list)
            self.features = output
        elif features is not None:
            if verbose:
                print('passing features through ...')
            output = features
            self.features = features
        elif self.features is not None:
            if verbose:
                print('using cached features ...')
            output = features
        else:
            raise ValueError('no features or data found')
        return output

    def predict(self, data=None, features=None, pred_contribs=False):
        from xgboost import DMatrix
        from scipy.sparse import vstack
        self.transform_features(data=data, features=features)
        if pred_contribs:
            contribs = list(self._feature_contribution_gen())
            return vstack(contribs)
        else:
            features = DMatrix(self.features, feature_names=self.feature_names)
            return self.model.predict(features)

    def _feature_contribution_gen(self):
        from xgboost import DMatrix
        from scipy.sparse import coo_matrix, csr_matrix
        features = csr_matrix(self.features)
        for i in range(features.shape[0]):
            dmat = DMatrix(features[i], feature_names=self.feature_names)
            output = self.model.predict(dmat, pred_contribs=True)
            yield coo_matrix(output)

    def _feature_contribution_summary_gen(self, n=5):
        from xgboost import DMatrix
        from scipy.sparse import csr_matrix
        features = csr_matrix(self.features)
        for i in range(features.shape[0]):
            temp_data = features[i]
            yield self._feature_summary(temp_data, n=n)

    def feature_contribution(self, data=None, features=None, n=5):
        self.transform_features(data=data, features=features)
        output = list(self._feature_contribution_summary_gen(n=n))
        return output

    def _fit_deciles(self, data, n=100, encode='ordinal', strategy='quantile'):
        import numpy as np
        from sklearn.preprocessing import KBinsDiscretizer
        data = np.reshape(data, (-1, 1))
        d = KBinsDiscretizer(n_bins=n, strategy=strategy, encode=encode)
        d.fit(data)
        self.deciles = d

    def _transform_deciles(self, data):
        import numpy as np
        from itertools import chain
        data = np.reshape(data, (-1, 1))
        deciles = self.deciles.transform(data)
        deciles = list(chain.from_iterable(deciles))
        return np.array(deciles, dtype=np.int) + 1

    def _feature_summary(self, feature, n=5):
        from xgboost import DMatrix
        import numpy as np
        import datetime
        dmat_feature = DMatrix(feature, feature_names=self.feature_names)
        dense_feature = np.reshape(feature.toarray(), (-1,))
        contrib = np.reshape(self.model.predict(dmat_feature, pred_contribs=True), (-1,))
        pred = self.model.predict(dmat_feature, pred_contribs=False)
        decile = self._transform_deciles(pred)[0]
        ranking = np.argsort(np.absolute(contrib)[0:len(dense_feature)])[::-1]
        output = '__________\r\n' + \
                 self.name + '\r\n' + \
                 'Prediction : ' + str(pred) + ' Percentile Rank: ' + str(decile) + '\r\n' + \
                 '. . . . . . . \r\n' + \
                 'Top Contributing Features: \r\n '
        for i in range(n):
            index = ranking[i]
            temp_name = self.feature_names[index]
            temp_feature_val = dense_feature[index]
            temp_feature_conrib = contrib[index]
            output = output + str(temp_name) + ': ' + str(temp_feature_val) + ', contribution: ' + \
                     str(temp_feature_conrib) + ' \r\n '
        output = output + 'Generation Date : ' + str(datetime.datetime.now()) + ' \r\n ' + 'Population Context Mean: ' + str(self.pred_mean)
        return output

    def save(self, file_name):
        import pickle
        import os
        self.features = None
        self.data = None
        self.preds = None
        pickle.dump(self, open(file_name, 'wb'))
        print('saved to ', os.getcwd() + '/' + file_name)

