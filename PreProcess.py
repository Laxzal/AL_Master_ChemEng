'''

Standardisation

'''
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


class MinMaxScaling(object):

    def __init__(self):
        self.scaler = None

    def minmaxscale(self, X_train, X_val, X_test):
        self.scaler = MinMaxScaler()
        # TODO Check whether I should pipeline this rather than manually
        X_train = self.scaler.fit_transform(X_train)
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
        self.scaler = StandardScaler()
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

