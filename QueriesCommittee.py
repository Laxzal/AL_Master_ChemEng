from collections import Counter

import numpy as np
from scipy.stats import entropy
from sklearn.exceptions import NotFittedError

import CommitteeClass
from Arguments import multi_argmax, shuffled_argmax


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
