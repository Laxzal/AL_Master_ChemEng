import abc
import sys
from typing import List, Iterator, Callable
from ToolsActiveLearning import retrieverows
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError

from BaseModel import BaseModel

if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta('ABC', (), {})


class Committee(ABC):

    def __init__(self, learner_list: List[BaseModel]):
        self.classes_ = None
        assert type(learner_list) == list
        self.learner_list = [learner_class() for learner_class in learner_list]

    def __len__(self) -> int:
        return len(self.learner_list)

    def __iter__(self) -> Iterator:
        for learner in self.learner_list:
            yield learner

    def printname(self):
        classifier_models = []
        for learner in self.learner_list:
            classifier_models.append(str(learner.model_type))
        return '_'.join(classifier_models)

    def print_list(self):
        for learner in self.learner_list:
            print(learner)

    def _set_classes(self):
        """
        Checks the known class labels by each learner, merges the labels and returns a mapping which maps the learner's
        classes to the complete label list.
        """
        # assemble the list of known classes from each learner
        try:
            # if estimators are fitted

            known_classes = tuple(learner.optimised_model.classes_ for learner in self.learner_list)
        except AttributeError:
            # handle unfitted estimators
            self.classes_ = None
            self.n_classes_ = 0
            return

        self.classes_ = np.unique(
            np.concatenate(known_classes, axis=0),
            axis=0
        )
        self.n_classes_ = len(self.classes_)

    def check_class_labels(self, *args: BaseEstimator):

        try:
            classes_ = [estimator.classes_ for estimator in args]
        except AttributeError:
            raise NotFittedError('Not all estimators are fitted. Fit all estimators before using this method')

    def gridsearch_commitee(self, X_train, y_train, c_weight, splits):
        for learner in self.learner_list:
            learner.gridsearch(X_train, y_train, c_weight, splits)

    def fit_data(self, X_train, y_train, **fit_kwargs):

        for learner in self.learner_list:
            learner.fit(X_train, y_train, **fit_kwargs)
        self._set_classes()
        return self

    def vote(self, X_val, **predict_kwargs):

        prediction = np.zeros(shape=(len(X_val), len(self.learner_list)))

        for learner_idx, learner in enumerate(self.learner_list):
            predict_val = learner.predict(X_val, **predict_kwargs)
            prediction[:, learner_idx] = predict_val

        return prediction

    def vote_proba(self, X_val, **predict_proba_kwargs):
        n_samples = X_val.shape[0]
        n_learns = len(self.learner_list)
        proba = np.zeros(shape=(n_samples, n_learns, self.n_classes_))

        self.check_class_labels(*[learner.optimised_model for learner in self.learner_list])

        # known class labels are the same for each learner
        # probability prediction is straightforward

        for learner_idx, learner in enumerate(self.learner_list):
            proba[:, learner_idx, :] = learner.predict_proba(X_val, **predict_proba_kwargs)

        return proba

    def query(self, committee: BaseModel, query_strategy: Callable, X_val, *query_args, **query_kwargs):

        query_result = query_strategy(committee, X_val, *query_args, **query_kwargs)
        query_result = tuple((query_result, retrieverows(X_val, query_result)))
        return query_result
