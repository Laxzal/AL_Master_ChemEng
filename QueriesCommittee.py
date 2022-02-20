from collections import Counter
from typing import Callable, Union

import numpy as np
from scipy.stats import entropy
from sklearn.exceptions import NotFittedError
from random import shuffle

from sklearn.metrics import pairwise_distances

import CommitteeClass
from Arguments import multi_argmax, shuffled_argmax
from pytorch_clysters import CosineClusters
from scipy.spatial.distance import cosine, euclidean


def vote_entropy(committee: CommitteeClass, X_val, **predict_proba_kwargs):
    n_learners = len(committee)

    try:
        votes = committee.vote(X_val)
    except NotFittedError:
        return print('There was an error in the vote function from CommitteeClass')

    p_vote = np.zeros(shape=(X_val.shape[0], len(committee)))

    for vote_idx, vote in enumerate(votes):
        vote_counter = Counter(vote)

        for class_idx, class_label in enumerate(committee.classes_):
            p_vote[vote_idx, class_idx] = vote_counter[class_label] / n_learners

    entr = entropy(p_vote, axis=1)
    return entr


def vote_entropy_sampling(committee: CommitteeClass, X_val, n_instances: int = 1,
                          random_tie_break=False, **predict_proba_kwargs):
    disagreement = vote_entropy(committee, X_val, **predict_proba_kwargs)

    if not random_tie_break:
        return multi_argmax(disagreement, n_instances=n_instances)
    return shuffled_argmax(disagreement, n_instances=n_instances)


def KLMaxDisagreement(committee: CommitteeClass, X_val, **predict_proba_kwargs):
    try:
        p_vote = committee.vote_proba(X_val)
    except NotFittedError:
        return print('There was an error in the vote_proba function from CommitteeClass')

    p_consensus = np.mean(p_vote, axis=1)

    learner_kl_div = np.zeros(shape=(X_val.shape[0], len(committee)))
    for learner_idx, _ in enumerate(committee):
        learner_kl_div[:, learner_idx] = entropy(np.transpose(p_vote[:, learner_idx, :]), qk=np.transpose(p_consensus))

    return np.max(learner_kl_div, axis=1)


def max_disagreement_sampling(committee: CommitteeClass, X_val, n_instances: int = 1,
                              random_tie_break=False, **disagreement_measure_kwargs):
    disagreement = KLMaxDisagreement(committee, X_val, **disagreement_measure_kwargs)

    if not random_tie_break:
        return multi_argmax(disagreement, n_instances=n_instances)

    return shuffled_argmax(disagreement, n_instances=n_instances)


def consensus_entropy(committee: CommitteeClass, X_val, **predict_kwargs):
    try:
        proba = committee.predict_proba(X_val, **predict_kwargs)
    except NotFittedError:
        print('Issue with predict proba of the committees')

    entr = np.transpose(entropy(np.transpose(proba)))

    return entr


def consensus_entropy_sampling(committee: CommitteeClass, X_val, n_instances: int = 1, random_tie_break=False,
                               **disagreement_measure_kwargs):
    disagreement = consensus_entropy(committee, X_val, **disagreement_measure_kwargs)

    if not random_tie_break:
        return multi_argmax(disagreement, n_instances=n_instances)

    return shuffled_argmax(disagreement, n_instances=n_instances)


def get_cluster_samples(data, num_clusters: int = 5, max_epoch: int = 5, limit: int = 5000):
    #if limit > 0:
    #    shuffle(data)
    #    data = data[:limit]

    cosine_clusters = CosineClusters(num_clusters)

    cosine_clusters.add_random_training_items(data)

    for i in range(0, max_epoch):
        print("Epoch " + str(i))
        added = cosine_clusters.add_items_to_best_cluster(data)

        if added == 0:
            break

        centroids = cosine_clusters.get_centroids()
        outliers = cosine_clusters.get_outliers()
        randoms = cosine_clusters.get_randoms(3, verbose=True)
        return centroids + outliers + randoms



def similarize_distance(distance_measure: Callable) -> Callable:


    def sim(*args, **kwargs):
        return 1/(1+distance_measure(*args, **kwargs))

    return sim


cosine_similarity = similarize_distance(cosine)
euclidean_similarity = similarize_distance(euclidean)

def information_density(X, metric: Union[str, Callable] = 'euclidean') -> np.ndarray:
    similarity_mtx = 1/(1+pairwise_distances(X, X, metric = metric))

    return similarity_mtx.mean(axis=1)