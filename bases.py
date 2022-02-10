"""
Base classes for Active Learning Algorithm
"""

import abc
import warnings
import sys
from typing import List, Callable, Iterator

from sklearn.base import BaseEstimator


class BaseModel(object):

    def __init__(self):
        pass

    def gridsearch(self):
        pass

    def fit_predict(self):
        pass


class BaseComittee(abc.ABC, BaseEstimator):

    def __init__(self, learner_list: List[BaseModel], query_strategy: Callable):
        assert type(learner_list) == list, 'learners must be supplied in a list'

        self.learner_list = learner_list
        self.query_strategy = query_strategy

    def __iter__(self) -> Iterator[BaseModel]:
        """Appears to iterate through the list of learners"""
        for learner in self.learner_list:
            yield learner

    def __len__(self) -> int:
        """Returns the number of learners - Why?"""
        return len(self.learner_list)

    def _fit_to_known(self, bootstrap: bool = False, **fit_kwargs):
        """ Fits all learners to the training data nd labels provided to it so far"""
        for learner in self.learner_list:
            learner._fit_to_known(bootstrap=bootstrap, **fit_kwargs)

    def _fit_on_new(self, X, y, bootstrap: bool = False, **fit_kwargs):

        for learner in self.learner_list:
            learner._fit_on_new(X, y, bootstrap=bootstrap, **fit_kwargs)

    def fit(self, classifier_opt, X_train, y_train, X_val, X_test, c_weight, **fit_kwargs) -> 'BaseComittee':
        for learner in self.learner_list:
            learner.fit_predict(classifier_opt, X_train, y_train, X_val, X_test, c_weight, **fit_kwargs)
        return self

    def query(self, X_val, return_metrics: bool = False, *query_args, **query_kwargs):

        try:
            query_result, query_metrics = self.query_strategy(
                self, X_val, *query_args, **query_kwargs)
        except:
            query_metrics = None
            query_result=self.query_strategy(self, X_val, *query_args,**query_kwargs)

        if return_metrics:
            if query_metrics is None:
                warnings.warn(''
                              'The selected query strategy does not support return_metrics')
                return query_result, retrieve_rows(X_val, query_result), query_metrics
        else:
            return query_result, retrieve_rows(X_val, query_result)

    def teach(self, X, y, bootstrap: bool=False, only_new: bool = False, **fit_kwargs):

        self._add_training_data(X, y)
        if not only_new:
            self._fit_to_known(bootstrap=bootstrap, **fit_kwargs)
        else:
            self._fit_on_new(X, y, bootstrap=bootstrap, **fit_kwargs)