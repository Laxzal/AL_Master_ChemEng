
'''

Train Model

'''
import time

import numpy as np
from sklearn import metrics
import scikitplot as scplt

class TrainModel:

    def __init__(self, model_object):
        self.proba = None
        self.optimised_model = None
        self.train_y_predicted = None
        self.val_y_predicted = None
        self.test_y_predicted = None
        self.run_time = None
        self.accuracies = []
        self.model_object = model_object()

    def print_model_type(self):
        print(self.model_object.model_type)

    # Train the model and get the probabilities of the validation set - We use the probabilities to select the most
    # uncertain set
    def optimise(self, X_train, y_train, c_weight, splits, scoring):
        t0 = time.time()
        self.optimised_model = self.model_object.gridsearch(X_train, y_train, c_weight, splits, scoring)
        self.run_time = time.time() - t0
        print("-------------------------------------")
        print("-----Time to complete GridSearch----- \n", self.run_time)
        print("-------------------------------------")
        return self.optimised_model

    def train(self, X_train, y_train, X_val, X_test):
        print('Train set:', X_train.shape, 'y:', y_train.shape)
        print('Val set: ', X_val.shape)
        print('Test set: ', X_test.shape)
        t0 = time.time()
        self.optimised_model = \
            self.model_object.fit(X_train, y_train)
        self.run_time = time.time() - t0
        return X_train, X_val, X_test

    def predictions(self, X_train, X_val, X_test):
        self.val_y_predicted = self.model_object.predict(X_val)
        self.train_y_predicted, self.test_y_predicted = self.model_object.predict_labelled(X_train, X_test)
        self.proba = self.model_object.predict_proba(X_val)
        return self.proba

    # def _set_classes(self):
    #   try:
    #      known_classes = tuple(self.optimised_classifier.classes_ for learner in self.learner_list)

    def return_accuracy(self, i, y_test, y_train):
        classif_rate = np.mean(self.test_y_predicted.ravel() == y_test.ravel()) * 100  # TODO
        self.accuracies.append(classif_rate)
        print('--------------------------------')
        print('Iteration:', i)
        print('--------------------------------')
        print('y-test set:', y_test.shape)
        print('y_predicted_test:', self.test_y_predicted.shape)
        print('Example run in %.3f s' % self.run_time, '\n')
        print("Accuracy rate for %f " % (classif_rate))
        print("Classification report for classifier(Y_test) %s:\n%s\n" % (
            self.model_object.optimised_model, metrics.classification_report(y_test, self.test_y_predicted)))
        print("Confusion matrix(y_test):\n%s" % metrics.confusion_matrix(y_test, self.test_y_predicted))
        print("Confusion matrix(y_test):\n%s" % scplt.metrics.plot_confusion_matrix(y_test, self.test_y_predicted,
                                                                                    title="Confusion Matrix (y_test)"))
        print('--------------------------------')
        print("Classification report for classifier(Y_train) %s:\n%s\n" % (
            self.model_object.optimised_model, metrics.classification_report(y_train, self.train_y_predicted)))
        print("Confusion matrix(y_train):\n%s" % metrics.confusion_matrix(y_train, self.train_y_predicted))
        print("Confusion matrix(y_train):\n%s" % scplt.metrics.plot_confusion_matrix(y_train, self.train_y_predicted,
                                                                                     title="Confusion Matrix (y_train)"))
        print('--------------------------------')