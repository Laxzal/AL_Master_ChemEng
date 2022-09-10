from typing import List, Union, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.special import softmax


def retrieverows(X_val, I: Union[int, List[int], np.ndarray]) -> Union[sp.csr_matrix, np.ndarray, pd.DataFrame]:
    if sp.issparse(X_val):

        try:
            return X_val[I]
        except:
            sp_format = X_val.getformat()
            return X_val.tocsr()[I].asformat(sp_format)
    elif isinstance(X_val, pd.DataFrame):
        return X_val.iloc[I]
    elif isinstance(X_val, np.ndarray):
        return X_val[I]

    raise TypeError('%s datatype is not supported' % type(X_val))


def data_hstack(blocks: Sequence):
    """
    Stack horizontally sparse/dense arrays and pandas data frames.
    Args:
        blocks: Sequence of modALinput objects.
    Returns:
        New sequence of horizontally stacked elements.
    """
    if any([sp.issparse(b) for b in blocks]):
        return sp.hstack(blocks)
    elif isinstance(blocks[0], pd.DataFrame):
        pd.concat(blocks, axis=1)
    elif isinstance(blocks[0], np.ndarray):
        return np.hstack(blocks)
    elif isinstance(blocks[0], list):
        return np.hstack(blocks).tolist()

    TypeError('%s datatype is not supported' % type(blocks[0]))

def data_shape(X):

    try:
        return X.shape
    except:
        if isinstance(X, list):
            return np.array(X).shape

    raise TypeError("%s datatype is not supported" % type(X))


def data_vstack(blocks: Sequence):

    if any([sp.issparse(b) for b in blocks]):
        return sp.vstack(blocks)
    elif isinstance(blocks[0], pd.DataFrame):
        return blocks[0].append(blocks[1:])
    elif isinstance(blocks[0], np.ndarray):
        return np.concatenate(blocks)
    elif isinstance(blocks[0], list):
        return np.concatenate(blocks).tolist()


    raise TypeError("%s datatype is not supported" % type(blocks[0]))


def print_feature_importance_shap_values(shap_values, features):
    '''

    Prints the feature importances based on the SHAP values in an ordered way
    :param shap_values: tje SJA{ va;ues calculated from a shap.Explainer object
    :param features: The name of the features, on the order presented to the explainer
    :return:
    '''
    # Calculates the feature importance (mean absolute shap value) for each feature
    importances = []
    for i in range(shap_values.values.shape[1]):
        importances.append(np.mean(np.abs(shap_values.values[:, i])))
    # Calculates the normalized version
    importances_norm = softmax(importances)
    # Organize the importances and columns in a dictionary
    feature_importances = {fea: imp for imp, fea in zip(importances, features)}
    feature_importances_norm = {fea: imp for imp, fea in zip(importances_norm, features)}
    # Sorts the dictionary
    feature_importances = {k: v for k, v in sorted(feature_importances.items(), key=lambda item: item[1], reverse = True)}
    feature_importances_norm= {k: v for k, v in sorted(feature_importances_norm.items(), key=lambda item: item[1], reverse = True)}
    # Prints the feature importances
    for k, v in feature_importances.items():
        print(f"{k} -> {v:.4f} (softmax = {feature_importances_norm[k]:.4f})")