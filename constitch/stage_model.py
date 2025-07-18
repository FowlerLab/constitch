import collections
import math
import dataclasses
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model
import sklearn.mixture
import sklearn.base

from . import alignment
from .constraints import Constraint

class ConversionStageModel(sklearn.base.BaseEstimator):
    def __init__(self, model=None):
        self.model = model or sklearn.linear_model.LinearRegression(fit_intercept=False)

    def conversion_func(self, poses1, poses2):
        raise NotImplementedError

    def split_X(self,X):
        return X[:,:X.shape[1]//2], X[:,X.shape[1]//2:]

    def fit(self, X, y):
        X = np.asarray(X)
        X = self.conversion_func(*self.split_X(X))
        return self.model.fit(X, y)

    def predict(self, X):
        X = np.asarray(X)
        X = self.conversion_func(*self.split_X(X))
        return self.model.predict(X)

    def fit_predict(self, X, y):
        X = np.asarray(X)
        X = self.conversion_func(*self.split_X(X))
        return self.model.fit_predict(X, y)

    def score(self, X, y):
        X = np.asarray(X)
        X = self.conversion_func(*self.split_X(X))
        return self.model.score(X, y)

    #def __getattr__(self, attr):
        #return getattr(self.model, attr)

    def __repr__(self):
        return type(self).__name__ + "(model={})".format(repr(self.model))

    def __str__(self):
        return type(self).__name__ + "(model={})".format(str(self.model))


class SimpleOffsetModel(ConversionStageModel):
    """ Simple stage model that calculates the offsets of the two
    images and gives them to the internal model (default LinearRegression)
    for estimation.

    This model is the most simple, it only considers the relative positions of
    the two images
    """
    def conversion_func(self, poses1, poses2):
        return poses2 - poses1


class GlobalStageModel(ConversionStageModel):
    """ Stage model that takes into account the global positions of the
    two images, by passing both image positions to the model unchanged.

    This is equivalent to just using the internal model directly, this is
    just a separate class for documentation and consistency.
    """
    def __init__(self, model=None):
        model = model or sklearn.linear_model.LinearRegression(fit_intercept=True)
        super().__init__(model)

    def conversion_func(self, poses1, poses2):
        return np.concatenate([poses1, poses2], axis=1)


class StageModelAligner(alignment.Aligner):
    def __init__(self, model, error=15, score=0.2):
        self.model = model
        self.error = error
        self.score = score

    def align(self, constraint, precalc1=None, precalc2=None):
        X = np.array([*constraint.box1.position, *constraint.box2.position]).reshape(1,-1)
        y = self.model.predict(X).reshape(-1)
        newconst = Constraint(constraint, dx=y[0], dy=y[1], score=self.score, error=self.error)
        return newconst

