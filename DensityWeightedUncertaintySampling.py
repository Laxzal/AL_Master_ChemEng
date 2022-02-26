import os

import numpy as np
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
import pandas as pd

from Arguments import shuffled_argmax, multi_argmax

"""
1. Cluster the data
2. Estimate P(y|k)
3. Calculate P(y|x)
4. Choose unlabeled sample based on Eq1 and label
5. Re-cluster if necessary
6. Repeat steps until stop
"""
wrk_path_3 = r"/Users/calvin/Documents/OneDrive/Documents/2020/Liposomes Vitamins/LiposomeFormulation"
os.chdir(wrk_path_3)


def unlabelled_data(file, method):
    ul_df = pd.read_csv(file)
    column_drop = ['Duplicate_Check',
                   'PdI Width (d.nm)',
                   'PdI',
                   'Z-Average (d.nm)',
                   'ES_Aggregation']
    ul_df = ul_df.drop(columns=column_drop)
    ul_df.replace(np.nan, 'None', inplace=True)
    ul_df = pd.get_dummies(ul_df, columns=["Component_1", "Component_2", "Component_3"],
                           prefix="", prefix_sep="")
    # if method=='fillna':    ul_df['Component_3'] = ul_df['Component_3'].apply(lambda x: None if pd.isnull(x) else x) #TODO This should be transformed into an IF function, thus when the function for unlabelled is filled with a parameter, then activates

    ul_df = ul_df.groupby(level=0, axis=1, sort=False).sum()

    print(ul_df.isna().any())
    X_val = ul_df.to_numpy()
    columns_x_val = ul_df.columns
    return X_val, columns_x_val


file = "stratified_sample_experiment.csv'"

X_val, columns_x_val = unlabelled_data('stratified_sample_experiment.csv',
                                       method='fillna')


class DensityWeightedUncertaintySampling:

    def __init__(self, X_uld: np.ndarray, X_lld: np.ndarray, y_labelled: np.ndarray, sigma: int = 25,
                 clusters: int = 10, max_iter: int = 100, C: float = 0.1, randomize_tie_break: bool = False,
                 n_instances: int = 10, ):
        self.P_x_k = None
        self.d = None
        self.distrib_array = None
        self.centres_ck = None
        self.all_x = None
        self.all_y = None
        self.C = C
        self.sigma = sigma
        self.clusters = clusters
        self.max_iter = max_iter
        # Unlabelled Data
        self.X_uld = X_uld

        self.kmeans = KMeans(n_clusters=self.clusters, random_state=42)

        # Initial P(k) Value
        self.P_k = np.divide(np.ones(self.clusters), float(clusters))

        self.P_k_x = None
        self.P_x = None
        # Labelled Data
        self.X_lld = X_lld

        self.y_labelled = y_labelled
        self.y_labelled_entries = np.array([i for i in self.y_labelled if i]).reshape(-1, 1)
        self.all_x_y()

        self.kmeans_fit()
        self.randomize_tie_break = randomize_tie_break
        self.n_instances = n_instances

    def all_x_y(self):
        y_unlabelled_list = [None] * self.X_uld.shape[0]

        self.all_y = np.concatenate((y_unlabelled_list, self.y_labelled), axis=0)
        self.all_x = np.concatenate((self.X_uld, self.X_lld), axis=0)
        self.d = len(self.all_x[0])
        self.P_x_k = np.zeros((len(self.all_x), self.clusters))

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

    def kmeans_fit(self):
        self.kmeans.fit(self.all_x)
        self.centres_ck = self.kmeans.cluster_centers_

        return self

    def distribution(self):

        # Empty Distribution Array to be filler: exp^(-1*(||x-ck||)/2/sigma^2)
        self.distrib_array = np.zeros((len(self.all_x), self.clusters))
        for i in range(self.clusters):
            self.distrib_array[:, i] = np.exp((-1 * (np.abs(self.all_x - self.centres_ck[i]) ** 2).sum(axis=1))
                                              / (2 * np.square(self.sigma)))
        return self.distrib_array

    def em_step(self):
        self.distribution()
        for _ in range(self.max_iter):
            # E-Step P(k|x)
            temp = self.distrib_array * np.tile(self.P_k, (len(self.all_x), 1))
            P_k_x = np.divide(temp, np.tile(np.sum(temp, axis=1), (self.clusters, 1)).T)
            # M_step
            self.P_k = 1. / (len(self.all_x) * np.sum(P_k_x, axis=0))
        self.P_k_x = P_k_x
        return self.P_k, self.P_k_x

    def prob_x_k(self):
        for i in range(self.clusters):
            self.P_x_k[:, i] = multivariate_normal.pdf(
                x=self.all_x,
                mean=self.centres_ck[i],
                cov=np.ones(self.d) * self.sigma,
            )
        return self.P_x_k

    def prob_x(self):
        self.em_step()
        self.prob_x_k()

        self.P_x = np.dot(self.P_x_k, self.P_k).reshape(-1)

        return self.P_x

    def make_query(self):
        unlabeled_entry_ids, _ = self.unlabelled_entries_ids()
        labeled_entry, labels = self.labeled_entries()
        labeled_entry_ids = self.labeled_entries_id()

        centres = self.centres_ck
        labels = np.asarray(labels).reshape(-1, 1)

        P_k_x = self.P_k_x
        p_x = self.P_x[list(unlabeled_entry_ids)]

        clf = DensityWeightedLogisticRegression(P_k_x[labeled_entry_ids, :],
                                                centres,
                                                C=self.C)
        clf.train(labeled_entry_ids, labels)

        P_y_k = clf.predict()
        P_y_x = np.zeros(len(unlabeled_entry_ids))

        for k, centre in enumerate(centres):
            P_y_x += P_y_k[k] * P_k_x[unlabeled_entry_ids, k]

        # binary case
        expected_error = P_y_x
        expected_error[P_y_x >= 0.5] = 1. - P_y_x[P_y_x >= 0.5]

        if not self.randomize_tie_break:
            query_idx = multi_argmax((expected_error * p_x), self.n_instances)
        else:
            query_idx = shuffled_argmax((expected_error * p_x), self.n_instances)

        return unlabeled_entry_ids[query_idx]


"""

formula used in the github libact
for i in range(clusters):
    empty_array[:,i] = np.exp(-np.einsum('ij,ji->i',(X_val-centres_ck[i]),(X_val-centres_ck[i]).T)/2/sigma)
    

"""


class DensityWeightedLogisticRegression(object):

    def __init__(self, density_estimate, centres, C: float = 0.01):
        self.density = np.asarray(density_estimate)
        self.centres = np.asarray(centres)
        self.C = C
        self.w_ = None

    def _likelihood(self, w, X, y):
        w = w.reshape(-1, 1)
        sigmoid = lambda t: 1. / (1. + np.exp(-t))

        L = lambda w: (self.C / 2. * np.dot(w[:-1].T, w[:-1]) -
                       np.sum(np.log(
                           np.sum(self.density *
                                  sigmoid(np.dot(y,
                                                 (np.dot(self.centres, w[:-1]) + w[-1]).T)
                                          ), axis=1)
                       ), axis=0))[0][0]

        return L(w)

    def train(self, X, y):
        d = np.shape(self.centres)[1]
        w = np.zeros((d + 1, 1))

        result = minimize(lambda _w: self._likelihood(_w, X, y),
                          w.reshape(-1),
                          method="CG")

        w = result.x.reshape(-1, 1)
        self.w_ = w
        return self.w_

    def predict(self):

        if self.w_ is not None:
            sigmoid = lambda t: 1. / (1. + np.exp(-t))
            return sigmoid(np.dot(self.centres, self.w_[:-1]) + self.w_[-1])
        else:
            print("The model is not trained")
            pass
