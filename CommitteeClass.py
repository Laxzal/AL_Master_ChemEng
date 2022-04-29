import abc
import sys
from typing import List, Iterator, Callable
import pickle
from datetime import datetime

import lime.lime_tabular
from matplotlib import pyplot as plt

from ToolsActiveLearning import retrieverows
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.metrics import *
from BaseModel import BaseModel

if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta('ABC', (), {})

plt.rcParams.update({'figure.figsize': (7, 5), 'figure.dpi': 100})


class CommitteeClassification(ABC):

    def __init__(self, learner_list: List[BaseModel], X_training, X_testing, y_training, y_testing, X_unlabeled,
                 query_strategy: Callable, c_weight=None, splits: int = 5,
                 scoring_type: str = 'precision', kfold_shuffle: bool = True):
        assert scoring_type in ['accuracy', 'balanced_accuracy', 'top_k_accuracy', 'average_precision',
                                'neg_brier_score',
                                'f1', 'f1_micro', 'f1_macro', 'f1_weighted', 'f1_samples', 'neg_los_loss', 'precision',
                                'recall', 'jaccard', 'roc_auc', 'roc_auc_ovr', 'roc_auc_ovo', 'roc_auc_ovr_weighted',
                                'roc_auc_ovo_weighted']

        self.classes_ = None
        assert type(learner_list) == list
        self.learner_list = [learner_class() for learner_class in learner_list]
        self.X_training = X_training
        self.X_testing = X_testing
        self.y_training = y_training
        self.y_testing = y_testing
        self.X_unlabeled = X_unlabeled
        self.c_weight = c_weight
        self.splits = splits
        self.query_strategy = query_strategy
        self.scoring_type = scoring_type
        self.kfold_shuffle = kfold_shuffle

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

    def gridsearch_committee(self, grid_params=None):
        # score_values = np.zeros(shape=[len(self.learner_list), len(self.learner_list), len(self.learner_list)])

        score_values = {}
        for learner_idx, learner in enumerate(self.learner_list):
            # score_values[:, :, learner_idx] \
            score_values[learner.model_type] = learner.gridsearch(X_train=self.X_training, y_train=self.y_training,
                                                                  c_weight=self.c_weight,
                                                                  catboost_weight=np.unique(self.y_training),
                                                                  splits=self.splits,
                                                                  scoring_type=self.scoring_type,
                                                                  kfold_shuffle=self.kfold_shuffle,
                                                                  parameters = grid_params[str(learner.model_type)])
        return score_values

    def fit_data(self, **fit_kwargs):

        for learner in self.learner_list:
            learner.fit(X_train=self.X_training, y_train=self.y_training, **fit_kwargs)
        self._set_classes()
        return self

    def vote(self, **predict_kwargs):

        prediction = np.zeros(shape=(len(self.X_unlabeled), len(self.learner_list)))

        for learner_idx, learner in enumerate(self.learner_list):
            predict_val = learner.predict(self.X_unlabeled, **predict_kwargs)
            prediction[:, learner_idx] = predict_val

        return prediction

    def vote_proba(self, **predict_proba_kwargs):
        n_samples = self.X_unlabeled.shape[0]
        n_learns = len(self.learner_list)
        proba = np.zeros(shape=(n_samples, n_learns, self.n_classes_))

        self.check_class_labels(*[learner.optimised_model for learner in self.learner_list])

        # known class labels are the same for each learner
        # probability prediction is straightforward

        for learner_idx, learner in enumerate(self.learner_list):
            proba[:, learner_idx, :] = learner.predict_proba(self.X_unlabeled, **predict_proba_kwargs)

        return proba

    def query(self, committee: BaseModel, *query_args, **query_kwargs):

        query_result = self.query_strategy(committee, self.X_unlabeled, *query_args, **query_kwargs)
        query_result = tuple((query_result, retrieverows(self.X_unlabeled, query_result)))
        return query_result

    def save_model(self):

        # TODO Set save location
        # TODO Add score and other details [maybe a dataframe]
        today_date = datetime.today().strftime('%Y%m%d')
        for learner_idx, learner in enumerate(self.learner_list):
            filename = str(learner.model_type) + '_committee_' + str(today_date) + '.sav'
            pickle.dump(learner, open(filename, 'wb'))

    def load_model(self, list_files: List):
        list_files = list_files
        # TODO Need to see if it works, may need to code in prediction functions
        for learner_idx, learner in enumerate(self.learner_list):
            loaded_model = pickle.load(open(list_files[learner_idx], 'rb'))
            learner[learner_idx] = loaded_model

    def confusion_matrix(self):
        # fig, axes = plt.subplots(1,len(self.learner_list))
        conf_dict = {}
        for learner_idx, learner in enumerate(self.learner_list):
            learner.predict_labelled(self.X_training, self.X_testing)
            cm_test, cm_train = learner.confusion_matrix(y_test=self.y_testing, y_train=self.y_training,
                                                         class_estim=self.classes_)
            conf_dict[str(learner.model_type) + '_train'] = cm_train
            conf_dict[str(learner.model_type) + '_test'] = cm_test
        return conf_dict

    def precision_scoring(self):
        scoring = {}
        for learner_idx, learner in enumerate(self.learner_list):
            if learner.train_y_predicted is None:
                learner.predict_labelled(self.X_training, self.X_testing)
            score_train, score_test = learner.precision_score_model(y_train = self.y_training, y_test = self.y_testing)
            scoring[str(learner.model_type) + '_train'] = score_train
            scoring[str(learner.model_type) + '_test'] = score_test

        return scoring

    def lime_analysis(self, feat_names, target_names):

        explainer = lime.lime_tabular.LimeTabularExplainer(self.X_training, feature_names=feat_names,
                                                           class_names=target_names,
                                                           discretize_continuous=True)
        #Explaining the instances

        i = np.random.randint(0, self.X_testing.shape[0])
        for learner_idx, learner in enumerate(self.learner_list):
            exp = explainer.explain_instance(self.X_testing[i], learner.test_y_predicted, num_features=len(feat_names))
            exp.sh

        #Extract the Feature Names
        #Get Class Names
        #Get Labels
        #Get the Categorical Features
        pass


class CommitteeRegressor(ABC):

    def __init__(self, learner_list: List[BaseModel], X_training, X_testing, y_training, y_testing, X_unlabeled,
                 query_strategy: Callable,
                 splits: int = 5, kfold_shuffle: bool = True, scoring_type: str = 'r2', instances: int = 10):

        self.score_parameters = {'r2': r2_score, 'explained_variance': explained_variance_score, 'max_error': max_error,
                                 'neg_mean_absolute_error': mean_absolute_error,
                                 'neg_mean_squared_error': mean_squared_error,
                                 'neg_root_mean_squared_error': mean_squared_error,
                                 'neg_mean_squared_log_error': mean_squared_log_error,
                                 'neg_median_absolute_error': median_absolute_error,
                                 'neg_mean_poisson_deviance': mean_poisson_deviance,
                                 'neg_mean_gamma_deviance': mean_gamma_deviance,
                                 'neg_mean_absolute_percentage_error': mean_absolute_percentage_error}

        assert scoring_type in list(self.score_parameters.keys())

        self.classes_ = None
        assert type(learner_list) == list
        self.learner_list = [learner_class() for learner_class in learner_list]
        self.X_training = X_training
        self.X_testing = X_testing
        self.y_training = y_training
        self.y_testing = y_testing
        self.X_unlabeled = X_unlabeled
        self.scoring_type = scoring_type
        self.splits = splits
        self.kfold_shuffle = kfold_shuffle
        self.instances = instances
        self.query_strategy = query_strategy

        self.score_query = self.score_parameters[self.scoring_type]

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

    def gridsearch_committee(self):
        score_values = {}
        for learner_idx,  learner in enumerate(self.learner_list):
            score_values[learner.model_type] = learner.gridsearch(X_train=self.X_training, y_train=self.y_training, splits=self.splits,
                               kfold_shuffle=self.kfold_shuffle, scoring_type=self.scoring_type)

        return score_values

    def fit_data(self, **fit_kwargs):

        for learner in self.learner_list:
            learner.fit(self.X_training, self.y_training, **fit_kwargs)
        self._set_classes()
        return self

    def predict(self, X, return_std: bool = False, **predict_kwargs):

        vote = self.vote(X, **predict_kwargs)
        if not return_std:
            return np.mean(vote, axis=1)
        else:
            return np.mean(vote, axis=1), np.std(vote, axis=1)

    def vote(self, X, **predict_kwargs):

        prediction = np.zeros(shape=(len(X), len(self.learner_list)))

        for learner_idx, learner in enumerate(self.learner_list):
            prediction[:, learner_idx] = learner.predict(X, **predict_kwargs).reshape(-1, )

        return prediction

    def query(self, committee: BaseModel, *query_args, **query_kwargs):

        query_result = self.query_strategy(committee, self.X_unlabeled, *query_args, **query_kwargs)
        query_result = tuple((query_result, retrieverows(self.X_unlabeled, query_result)))
        return query_result

    def score(self, **predict_kwargs):

        train_vote = self.vote(self.X_training, **predict_kwargs)
        test_vote = self.vote(self.X_testing, **predict_kwargs)
        scores = {}

        for learner_idx, learner in enumerate(self.learner_list):
            train_strat = self.score_query(self.y_training, train_vote[:, learner_idx])
            print("X training data scoring")
            print("Model: ", learner.model_type)
            print("Scoring Strategy: ", str(self.scoring_type))
            print("Score: ", train_strat)
            scores[str(learner.model_type) + '_train'] = np.array([str(self.scoring_type), train_strat])
        for learner_idx, learner in enumerate(self.learner_list):
            test_strat = self.score_query(self.y_testing, test_vote[:, learner_idx])
            print("X testing data scoring")
            print("Model: ", learner.model_type)
            print("Scoring Strategy: ", str(self.scoring_type))
            print("Score: ", test_strat)
            scores[str(learner.model_type) + '_test'] = np.array([str(self.scoring_type), test_strat])

        return scores
