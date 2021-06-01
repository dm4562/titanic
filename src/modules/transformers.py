from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

import pandas as pd


class OneHotTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, col=None, **kwargs):
        super().__init__()
        self.enc = OneHotEncoder(**kwargs)
        self.col = col

    def fit(self, X, y=None):
        assert isinstance(X, pd.DataFrame)

        temp = X[self.col].to_numpy().reshape(-1, 1)
        self.enc.fit(temp, y)
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)

        X_ = X.copy()
        temp = X_[self.col].to_numpy().reshape(-1, 1)
        temp = self.enc.transform(temp)
        categories = [f"{self.col}_{c}" for c in self.enc.categories_[0][1::]]

        df = pd.DataFrame(data=temp, columns=categories)
        for x in categories:
            X_[x] = df[x]

        return X_


class ImputeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, col=None, **kwargs):
        super().__init__()
        self.imp = SimpleImputer(**kwargs)
        self.col = col

    def fit(self, X, y=None):
        assert isinstance(X, pd.DataFrame)

        temp = X[self.col].to_numpy().reshape(-1, 1)
        self.imp.fit(temp, y)
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)

        X_ = X.copy()
        temp = X_[self.col].to_numpy().reshape(-1, 1)
        temp = self.imp.transform(temp)

        temp = pd.DataFrame(data=temp, columns=[self.col])
        X_[self.col] = temp[self.col]
        return X_


class SmoothMeanTransformer(BaseEstimator, TransformerMixin):
    """
    This class encodes columns with very high cardinality. One-hot
    encoding would make the number of features very high. In this
    encoding we need access to the output value, we calculate base
    `prior` probability of y. We then calculate the probability of y
    for each entry in the column.

    we then calculate the smooth value as
    (counts * means + m * prior) / (counts + m)
    which is a weighted avg of the prior and the individual prob of
    each entry.
    """

    def __init__(self, col=None, m=150):
        super().__init__()
        self.col = col
        self.prior = None
        self.smooth = None
        self.m = m

    def fit(self, X, y=None):
        assert isinstance(X, pd.DataFrame)

        self.prior = X[self.col].mean()
        y_label = "__y"

        X[y_label] = y

        # Compute the number of values and the mean of each group
        train_agg = X.groupby(self.col)[y_label].agg(['count', 'mean'])
        counts = train_agg['count']
        means = train_agg['mean']

        # Compute the "smoothed" means for train dataset
        self.smooth = (counts * means + self.m * self.prior) / (counts + self.m)

        X.drop(columns=y_label)
        return self

    def transform(self, X, y):
        assert isinstance(X, pd.DataFrame)
        assert self.prior

        X_ = X.copy()
        X_[self.col]

        # Entries that dont exist in self.smooth will just get overwritten
        # by prior
        X_[self.col] = X_[self.col].map(self.smooth).fillna(self.prior)
        return X_
