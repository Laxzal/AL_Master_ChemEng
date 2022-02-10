from typing import List, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp


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
