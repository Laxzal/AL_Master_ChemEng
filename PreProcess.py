'''

Standardisation

'''
import numpy as np
from sklearn import preprocessing

from sklearn.decomposition import PCA

class MinMaxScaling(object):
    name = 'MinMax_Scaler'
    def __init__(self):
        self.scaler = preprocessing.MinMaxScaler()

    def fit_scale(self, X_train, X_test, X_val, converted_columns):
        temp_stack = np.vstack((X_train, X_test, X_val))
        # TODO
        #numerical_data = temp_stack[:, [not elem for elem in converted_columns]]
        #categorical_data = temp_stack[:, converted_columns]
        self.scaler.fit(temp_stack)

    def transform_scale(self, X_train, X_val, X_test, converted_columns):
        # TODO Check whether I should pipeline this rather than manually
        # numerical_x_train = X_train[:, [not elem for elem in converted_columns]]
        # categorical_x_train = X_train[:, converted_columns]
        #numerical_x_train_transformed = self.scaler.transform(numerical_x_train)

        #X_train = np.concatenate((numerical_x_train_transformed, categorical_x_train), axis=1)
        X_train = self.scaler.transform(X_train)

        X_val = self.scaler.transform(X_val)

        X_test = self.scaler.transform(X_test)


        return X_train, X_val, X_test

    def inverse(self, X_train, X_val, X_test, converted_columns):

        X_train = self.scaler.inverse_transform(X_train)
        X_val = self.scaler.inverse_transform(X_val)
        X_test = self.scaler.inverse_transform(X_test)



        return X_train, X_val, X_test


class Standardisation(object):
    name = 'Standardisation'
    def __init__(self):
        self.scaler = preprocessing.StandardScaler()


    def fit_scale(self, X_train, X_test, X_val, converted_columns):
        temp_stack = np.vstack((X_train, X_test, X_val))
        #TODO
        numerical_data = temp_stack[:, [not elem for elem in converted_columns]]
        categorical_data = temp_stack[:, converted_columns]
        self.scaler.fit(numerical_data)

        return self.scaler


    def transform_scale(self, X_train, X_val, X_test, converted_columns):
        # TODO Check whether I should pipeline this rather than manually
        numerical_x_train = X_train[:, [not elem for elem in converted_columns]]
        categorical_x_train = X_train[:, converted_columns]
        numerical_x_train_transformed = self.scaler.transform(numerical_x_train)
        X_train = np.concatenate((numerical_x_train_transformed, categorical_x_train), axis=1)

        numerical_x_val = X_val[:, [not elem for elem in converted_columns]]
        categorical_x_val = X_val[:, converted_columns]
        numerical_x_val_transformed = self.scaler.transform(numerical_x_val)
        X_val = np.concatenate((numerical_x_val_transformed, categorical_x_val), axis=1)

        numerical_x_test = X_test[:, [not elem for elem in converted_columns]]
        categorical_x_test = X_test[:, converted_columns]
        numerical_x_test_transformed = self.scaler.transform(numerical_x_test)
        X_test = np.concatenate((numerical_x_test_transformed, categorical_x_test), axis=1)

        return X_train, X_val, X_test

    def inverse(self, X_train, X_val, X_test, converted_columns):
        numerical_x_train = X_train[:, [not elem for elem in converted_columns]]
        categorical_x_train = X_train[:, converted_columns]
        numerical_x_train_transformed = self.scaler.inverse_transform(numerical_x_train)
        X_train = np.concatenate((numerical_x_train_transformed, categorical_x_train), axis=1)

        numerical_x_val = X_val[:, [not elem for elem in converted_columns]]
        categorical_x_val = X_val[:, converted_columns]
        numerical_x_val_transformed = self.scaler.inverse_transform(numerical_x_val)
        X_val = np.concatenate((numerical_x_val_transformed, categorical_x_val), axis=1)

        numerical_x_test = X_test[:, [not elem for elem in converted_columns]]
        categorical_x_test = X_test[:, converted_columns]
        numerical_x_test_transformed = self.scaler.inverse_transform(numerical_x_test)
        X_test = np.concatenate((numerical_x_test_transformed, categorical_x_test), axis=1)
        return X_train, X_val, X_test


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