import numpy as np


def multi_argmax(values, n_instances: int = 1):
    """
    Selects the indeices of the n_instances highest values
    :param values: contains the values to be selected from
    :param n_instances: specifies how many indices to return
    :return: the indices of the n_instances largest values
    """
    assert n_instances <= values.shape[0], 'n_instances must be less or equal than the size'

    max_idx = np.argpartition(-values, n_instances - 1, axis=0)[:n_instances]
    return max_idx


def shuffled_argmax(values, n_instances: int = 1):
    """
    Shuffles the values and sorts them afterwards. This can be used to break the tie when the highest score is not
    unique. The shuffle randomizes orderm which is preserved by the mergersort algorithm
    :param values: Contains the values to be selected from
    :param n_instances: specifies how many indices to return
    :return: the indices of the n_instances largest values
    """

    assert n_instances <= values.shape[0], 'n_instances must be less than or equal'

    shuffled_idx = np.random.permutation(len(values))
    shuffled_values = values[shuffled_idx]

    sorted_query_idx = np.argsort(shuffled_values, kind='mergesort')[len(shuffled_values) - n_instances:]

    query_idx = shuffled_idx[sorted_query_idx]

    return query_idx


def _is_arraylike(x):
    return (hasattr(x, '__len__') or
            hasattr(x, 'shape') or
            hasattr(x, '__array__'))


def nlargestarg(a, n):
    assert (_is_arraylike(a))
    assert (n > 0)

    argret = np.argsort(a)

    return argret[argret.size - n:]
