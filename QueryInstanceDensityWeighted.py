import numpy as np
from sklearn.metrics import pairwise_distances

from Arguments import nlargestarg


class QueryInstanceDensityWeighted:


    def __init__(self, X_uld: np.ndarray, X_lld: np.ndarray, y_labelled: np.ndarray, uncertainty_measure: str = 'entropy', distance: str = 'euclidean', beta: float = 1.0):

        self.d = None
        self.all_x = None
        self.all_y = None
        assert uncertainty_measure in ['least_confident', 'margin', 'entropy']
        assert distance in ['cityblock', 'cosine', 'euclidean','l1','l2','manhattan']
        self.X_uld = X_uld
        self.X_lld = X_lld
        self.y_labelled = y_labelled
        self.uncertainty_measure = uncertainty_measure
        self.distance = distance
        self.beta = beta
        self.all_x_y()

    def all_x_y(self):
        y_unlabeled_list = [None] * self.X_uld.shape[0]

        self.all_y = np.concatenate((y_unlabeled_list, self.y_labelled), axis = 0)
        self.all_x = np.concatenate((self.X_uld, self.X_lld), axis = 0)
        self.d = len(self.all_x[0])

        return self

    def mask_unlabelled(self):
        return np.fromiter((e is None for e in self.all_y), dtype=bool)

    def mask_labelled(self):
        return np.fromiter((e is not None for e in self.all_y), dtype=bool)

    def labeled_entries(self):
        return self.all_x[self.mask_labelled()], self.all_y[self.mask_labelled()].tolist()

    def labeled_entries_id(self):
        return np.where(self.mask_labelled())[0]

    def unlabelled_entries_ids(self):
        """
        :return:idx - numpy array, shape = (n_samples unlabeled)
                X   - numpy array, shape (n_sample unlabeled, n_features)
        """

        return np.where(self.mask_unlabelled())[0], self.all_x[self.mask_unlabelled()]


    def select(self, batch_size = 1, model = None, proba_prediction=None, **kwarg):
        _, unlab_feat =self.unlabelled_entries_ids()
        unlabel_index, _ = self.unlabelled_entries_ids()

        if model is None and proba_prediction is None:
            raise ValueError("Please provide model and proba predictions")
        elif model is not None and proba_prediction is None:
            pv = model.predict_proba(unlab_feat)
        else:
            assert len(proba_prediction) == len(unlab_feat)
            pv = np.asarray(proba_prediction)

        #Information
        spv = np.shape(pv)
        if self.uncertainty_measure == 'entropy':
            pat = [-np.sum(vec * np.log(vec+1e-9)) for vec in pv]
        elif self.uncertainty_measure =='margin':
            pat = np.partition(pv, (spv[1] - 2, spv[1] - 1), axis = 1)
            pat = pat[:, spv[1] - 2] - pat[:, spv[1] - 1]

        elif self.uncertainty_measure == 'least_confident':
            pat = np.partition(pv, spv[1] - 1, axis = 1)
            pat = 1 - pat[:, spv[1] - 1]
        else:
            raise ValueError("Parameter uncertainty_measure shhould be in ['least_confident', 'margin', 'entropy'")

        #repre
        dis_mat = pairwise_distances(X=unlab_feat, metric = self.distance)
        div = np.mean(dis_mat, axis = 0)
        div = div ** self.beta

        assert len(pat) == len(div) == len(unlab_feat)
        scores = np.multiply(pat, div)
        return np.asarray(unlabel_index)[nlargestarg(scores, batch_size)]