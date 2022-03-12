import numpy as np
from sklearn.exceptions import NotFittedError
from scipy.stats import entropy

from Arguments import multi_argmax, shuffled_argmax


class SelectionFunction:

    def __init__(self):
        pass

    def proba_uncertainty(self, proba):
        return 1 - np.max(proba, axis=1)

    def proba_margin(self, proba):
        if proba.shape[1] == 1:
            return np.zeros(shape=len(proba))
        part = np.partition(-proba, 1, axis=1)
        margin = -part[:, 0] + part[:, 1]

        return margin

    def proba_entropy(self, proba):

        return np.transpose(entropy(np.transpose(proba)))

    def classifier_uncertainty(self, classifier, X, **predict_proba_kwargs):
        try:
            classwise_uncertainty = classifier.predict_proba(X, **predict_proba_kwargs)
        except NotFittedError:
            return np.ones(shape=(X.shape[0],))

        uncertainty = self.proba_uncertainty(classwise_uncertainty)
        return uncertainty

    def classifier_margin(self, classifier, X, **predict_proba_kwargs):
        try:
            classwise_uncertainty = classifier.predict_proba(X, **predict_proba_kwargs)
        except NotFittedError:
            return np.zeros(shape=(X.shape[0],))

        # if classwise_uncertainty.shape[1] == 1:
        #     return np.zeros(shape=(classwise_uncertainty.shape[0],))
        #
        # part = np.partition(-classwise_uncertainty, 1, axis=1)
        # margin = -part[:, 0] + part[:, 1]

        return self.proba_margin(classwise_uncertainty)

    def classifier_entropy(self, classifier, X, **predict_proba_kwargs):
        try:
            classwise_uncertainty = classifier.predict_proba(X, **predict_proba_kwargs)
        except NotFittedError:
            return np.zeros(shape=(X.shape[0],))

        return self.proba_entropy(classwise_uncertainty)

    def uncertainty_sampling(self, classifier, X, n_instances: int = 1, random_tie_break: bool = False,
                             **uncertainty_measure_kwargs):
        uncertainty = self.classifier_uncertainty(classifier, X, **uncertainty_measure_kwargs)

        if not random_tie_break:
            query_idx = multi_argmax(uncertainty, n_instances=n_instances)
        else:
            query_idx = shuffled_argmax(uncertainty, n_instances=n_instances)

        return query_idx, X[query_idx]

    def margin_sampling(self, classifier, X, n_instances: int = 1, random_tie_break: bool = False,
                        **uncertainty_measure_kwargs):
        margin = self.classifier_margin(classifier, X, **uncertainty_measure_kwargs)

        if not random_tie_break:
            query_idx = multi_argmax(-margin, n_instances=n_instances)
        else:
            query_idx = shuffled_argmax(-margin, n_instances=n_instances)

        return query_idx, X[query_idx]

    def entropy_sampling(self, classifier, X, n_instances: int = 1, random_tie_break: bool = False,
                         **uncertainty_measure_kwargs):
        entropy = self.classifier_entropy(classifier, X, **uncertainty_measure_kwargs)

        if not random_tie_break:
            query_idx = multi_argmax(entropy, n_instances=n_instances)
        else:
            query_idx = shuffled_argmax(entropy, n_instances=n_instances)
        return query_idx, X[query_idx]