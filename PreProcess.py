'''

Standardisation

'''
import numpy as np
from sklearn import preprocessing

from sklearn.decomposition import PCA

class MinMaxScaling(object):

    def __init__(self):
        self.scaler = preprocessing.MinMaxScaler()

    def minmaxscale_fit(self, X_train, X_test, X_val):
        temp_stack = np.vstack((X_train, X_test, X_val))

        self.scaler.fit(temp_stack)


    def minmaxscale_trans(self, X_train, X_val, X_test):
        # TODO Check whether I should pipeline this rather than manually
        X_train = self.scaler.transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)
        return X_train, X_val, X_test

    def inverse_minmaxscale(self, X_train, X_val, X_test):
        X_train = self.scaler.inverse_transform(X_train)
        X_val = self.scaler.inverse_transform(X_val)
        X_test = self.scaler.inverse_transform(X_test)
        return X_train, X_val, X_test


class Standardisation(object):

    def __init__(self, X_train: np.ndarray = None, X_test: np.ndarray = None, X_unlabeled: np.ndarray = None):
        self.scaler = preprocessing.StandardScaler()
        self.X_train = X_train
        self.X_test = X_test
        self.X_unlabeled = X_unlabeled

    def standardise(self):
        X_train = self.scaler.fit_transform(self.X_train)
        X_test = self.scaler.fit_transform(self.X_test)
        X_unlabeled = self.scaler.fit_transform(self.X_unlabeled)

        return X_train, X_test, X_unlabeled

    def inverse_standardise(self):
        X_train = self.scaler.inverse_transform(self.X_train)
        X_test = self.scaler.inverse_transform(self.X_test)
        X_unlabeled = self.scaler.inverse_transform(self.X_unlabeled)

        return X_train, X_test, X_unlabeled


class PCA_scale():

    def __init__(self, n_components: int=2):
        self.n_components = n_components
        self.pca = None
    def pca_fit(self, data):

        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(X = data)
        return self.pca

    def pca_transform(self, data):

        principleComponents = self.pca.transform(X = data)

        return principleComponents

        for _, data in enumerate(samples):
            for _, data_x in enumerate(data):
                similarity_scores.append(data_x[0])
                index.append(data_x[1])
                data_info.append(data_x[2])