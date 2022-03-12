from typing import List, Union, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp
try:
    import torch
except:
    pass

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

    try:
        if torch.is_tensor(blocks[0]):
            return torch.cat(blocks)
    except:
        pass

    raise TypeError("%s datatype is not supported" % type(blocks[0]))