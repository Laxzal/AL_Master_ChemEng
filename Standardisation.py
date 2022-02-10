'''

Standardisation

'''
from sklearn.preprocessing import MinMaxScaler


class MinMaxScaling(object):

    def __init__(self):
        self.scaler = None

    def normalise(self, X_train, X_val, X_test):
        self.scaler = MinMaxScaler()
        # TODO Check whether I should pipeline this rather than manually
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)
        return X_train, X_val, X_test

    def inverse_normalise(self, X_train, X_val, X_test):
        X_train = self.scaler.inverse_transform(X_train)
        X_val = self.scaler.inverse_transform(X_val)
        X_test = self.scaler.inverse_transform(X_test)
        return X_train, X_val, X_test