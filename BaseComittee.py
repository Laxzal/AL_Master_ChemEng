import abc
from typing import Optional, Callable
import sys
from sklearn.base import BaseEstimator
from sklearn.ensemble._base import _BaseHeterogeneousEnsemble
from sklearn.pipeline import Pipeline

from ToolsActiveLearning import data_hstack

if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta('ABC', (), {})


class BaseLearner(ABC, BaseEstimator):

    def __init__(self,
                 estimator: BaseEstimator,
                 query_strategy: Callable,
                 X_training: Optional = None,
                 y_training: Optional = None,
                 on_transformed: bool = False,
                 **fit_kwargs):
        assert callable(query_strategy), 'Query Stratgey must be callable'

        self.estimator = estimator
        self.query_strategy = query_strategy
        self.on_transformed = on_transformed

        self.X_training = X_training
        self.y_training = y_training

    def transform_without_estimating(self, X):
        '''
        Transforms the data s supplied to the estimnator
        :param X:
        :return:
        '''

        Xt = []
        pipes = [self.estimator]

        if isinstance(self.estimator, _BaseHeterogeneousEnsemble):
            pipes = self.estimator.estimators_

        # Trandoms data with pipelines used by estimator
        for pipe in pipes:
            if isinstance(pipe, Pipeline):
                transformation_pipe = pipe.__class__(steps=[*pipe.steps[:-1], ('passthrough', 'passthrough')])
                Xt.append(transformation_pipe(X))

        if not Xt:
            return X

        return data_hstack(Xt)



    def _fit_new(self, **fit_kwargs):
        print('Training...', self.estimator)
        self.estimator.fit(self.X_training, self.y_training, **fit_kwargs)

        return self

    #@abc.abstractmethod
    #def fit(self, *args, **kwargs):
     #   pass
# class BaseComittee(ABC, BaseEstimator):

# def __init__(self, learner_list: List[BaseLearner]):
