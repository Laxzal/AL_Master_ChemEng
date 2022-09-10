from typing import List, Callable

import numpy as np
import pandas as pd

from BaseModel import BaseModel
from ToolsActiveLearning import retrieverows


class ActiveLearner():


    def __init__(self, estimator: List[BaseModel],
                 query_strategy: Callable,
                 X_data: pd.DataFrame,
                 y_data: pd.DataFrame):



class BayesianOptimizer():


    def __init__(self, estimator: List[BaseModel],
                 query_strategy: Callable,
                 X_data: pd.DataFrame,
                 y_data: pd.DataFrame):

        max_idx = np.argmax(y_data)
        self.X_max = retrieverows(X_data, max_idx)
        self.y_max = y_data[max_idx]


    def _set_max(self, ):