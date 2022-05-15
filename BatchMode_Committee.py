from typing import Union, Callable, Optional

import gower
import scipy.sparse as sp
from sklearn.metrics.pairwise import (pairwise_distances,
                                      pairwise_distances_argmin)
import numpy as np

from BaseModel import BaseModel
from QueriesCommittee import max_disagreement_sampling, max_std_sampling
from ToolsActiveLearning import data_shape, data_vstack, retrieverows
from SelectionFunctions import SelectionFunction


def select_cold_start_instance(X: np.ndarray, metric: Union[str, Callable], n_jobs: Union[int, None]):
    # Compute all pairwise distances in our unlabeled data and obtain the row-wise average for each of our records in X
    n_jobs = n_jobs if n_jobs else 1
    average_distance = np.mean(pairwise_distances(X, metric=metric, n_jobs=n_jobs), axis=0)

    # Isolate and return our best instance for labeling as the record with the least average distance
    best_coldstart_instance_index = np.argmin(average_distance)

    return best_coldstart_instance_index, X[best_coldstart_instance_index].reshape(1, -1)


def select_instance(X_training, X_pool, converted_columns, X_uncertainty: np.ndarray,
                    mask: np.ndarray,
                    metric: Union[str, Callable],
                    n_jobs: Union[int, None]):
    """
    Cost iteration strategy for selecting another record from our unlabeled records

    Given a set of labeled records (X_training) and unlabeled records (X_pool) with uncertainty scores (
    X_uncertainty)' we would like to identify the best instance in X_pool that best balances uncertainty and
    dissimilarity
    :param X_training:
    :param X_pool:
    :param X_uncertainty:
    :param mask:
    :param metric:
    :param n_jobs:
    :return:

    """

    X_pool_masked = X_pool[mask]

    # Extract tje number of labeled and unlabeled records
    n_labeled_records, *rest = X_training.shape
    n_unlabeled, *rest = X_pool_masked.shape

    # Determine our alpha paramater as |U| / (|U| + |D|).
    # Note: Due to the appending of X_training and removin of X_pool within 'ranked batch',
    # alpha is not fixed through the model's lifetime

    alpha = n_unlabeled / (n_unlabeled + n_labeled_records)



    if metric in ['gower']:
        distance_scores = gower.gower_matrix(X_pool_masked.reshape(n_unlabeled, -1),
                                             X_training.reshape(n_labeled_records, -1),
                                             cat_features=converted_columns).min(axis=1)
    # Compute the pairwise distance (and then similarity) scores from every unlabeled record
    # to every record in X_training. The result is an array of shape (n_samples, ).
    else:
        if n_jobs == 1 or n_jobs is None:
            _, distance_scores = pairwise_distances_argmin(X_pool_masked.reshape(n_unlabeled, -1),
                                                           X_uncertainty.reshape(n_labeled_records, -1),
                                                           metric=metric)

        else:
            distance_scores = pairwise_distances(X_pool_masked.reshape(n_unlabeled, -1),
                                                 X_training.reshape(n_labeled_records, -1),
                                                 metric=metric, n_jobs=n_jobs).min(axis=1)

    similarity_scores = 1 / (1 + distance_scores)

    # Compute the final scores, which are a balance between how dissimilar a given reocrd
    # is with the records in X_uncertainty and how uncertain we are about its class
    scores = alpha * (1 - similarity_scores) + (1 - alpha) * X_uncertainty[mask]

    # Isolate and return our best instance fo rlabeling as the one with the largest score.

    best_instance_index_in_unlabeled = np.argmax(scores)
    n_pool, *rest = X_pool.shape
    unlabeled_indices = [i for i in range(n_pool) if mask[i]]
    best_instance_index = unlabeled_indices[best_instance_index_in_unlabeled]
    mask[best_instance_index] = 0
    return best_instance_index, X_pool[[best_instance_index]], mask


def ranked_batch(classifier,
                 unlabeled,
                 X_training,
                 converted_columns,
                 uncertainty_scores: np.ndarray,
                 n_instances: int,
                 metric: Union[str, Callable],
                 n_jobs: Union[int, None]):
    """
    Query our top :n_instances: to request for labeling


    :param classifier:
    :param unlabeled:
    :param uncertainty_scores:
    :param n_instances:
    :param metric:
    :param n_jobs:
    :return:
    """

    # trainsform unlabeled data if needed

    # if classifier.on_transformed:
    #    unlabeled = classifier.transform_without_estimating(unlabeled)

    # if classifier.X_training is None:
    if X_training is None:
        best_coldstart_instance_index, labeled = select_cold_start_instance(X=unlabeled, metric=metric, n_jobs=n_jobs)
        instance_index_ranking = [best_coldstart_instance_index]
    elif data_shape(X_training)[0] > 0:
        labeled = X_training[:]
        instance_index_ranking = []

        # The maximum of number of records to sample
    ceiling = np.minimum(unlabeled.shape[0], n_instances) - len(instance_index_ranking)

    # mask for unlabeled initialised as transparent
    mask = np.ones(unlabeled.shape[0], bool)

    for _ in range(ceiling):
        print('Instance: ', _)
        # Recieve the instance and corresponding indec from our unlabeled copy that scores highest
        instance_index, instance, mask = select_instance(X_training=labeled, X_pool=unlabeled,
                                                         converted_columns=converted_columns,
                                                         X_uncertainty=uncertainty_scores, mask=mask,
                                                         metric=metric, n_jobs=n_jobs)

        # Add our instance we have considered for labeleing to ur labeled set. Although we don't know it's label
        # we want further iterations to consider the newly added instance so we do not query the same instance
        # redundantly
        labeled = data_vstack((labeled, instance))

        instance_index_ranking.append(instance_index)
    return np.array(instance_index_ranking), uncertainty_scores[np.array(instance_index_ranking)]


def query(committee: BaseModel, query_strategy, X, *query_args, **query_kwargs):
    query_result, query_score = query_strategy(committee, X, *query_args, **query_kwargs)
    query_result = tuple((query_result, retrieverows(X, query_result)))
    return query_result, query_score


def batch_sampling(models,
                   X,
                   X_labelled,
                   converted_columns,
                   query_type: Callable = max_std_sampling,
                   n_instances: int = 20,
                   metric: Union[str, Callable] = 'euclidean',
                   n_jobs: Optional[int] = None,
                   **uncertainty_measure_kwargs):
    query_instances = len(X)
    _, query_score = query(committee=models, query_strategy=query_type, X=X, n_instances=query_instances,
                           **uncertainty_measure_kwargs)
    return ranked_batch(models, unlabeled=X, X_training=X_labelled, uncertainty_scores=query_score,
                        n_instances=n_instances, metric=metric, n_jobs=n_jobs, converted_columns=converted_columns)
