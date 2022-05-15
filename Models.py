import os
import sys
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import confusion_matrix, precision_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV, KFold
from sklearn.svm import SVC, SVR, LinearSVR
import catboost as cb
from sklearn.utils import compute_class_weight

from BaseModel import BaseModel
from MSVR import MSVR


class SvmModel(BaseModel):
    model_type = 'SVC'
    model_category = 'Classification'

    def __init__(self):
        self.paramgrid = None
        self.proba = None
        self.optimised_model = None
        self.val_y_predicted = None
        self.test_y_predicted = None
        self.stratifiedkfold = None
        self.deopt_classifier = None
        self.train_y_predicted = None

    def gridsearch(self, X_train, y_train, c_weight, catboost_weight: Optional[None], splits: int = 5,
                   scoring_type: str = 'precision',
                   kfold_shuffle: bool = True,
                   parameters = None):
        print("GridSearching SVM...")
        self.deopt_classifier = SVC(class_weight=c_weight, random_state=42, probability=True)
        self.stratifiedkfold = StratifiedKFold(n_splits=splits, shuffle=kfold_shuffle, random_state=42)
        # TODO Better define params

        # TODO Create kFold&Scoring PARAM choose
        self.svm_grid = GridSearchCV(self.deopt_classifier, param_grid=parameters, cv=self.stratifiedkfold, refit=True,
                                     n_jobs=-1,
                                     verbose=10,
                                     scoring=scoring_type)
        self.svm_grid.fit(X_train, y_train)

        print("Best Estimator: \n{}\n".format(self.svm_grid.best_estimator_))
        print("Best Parameters: \n{}\n".format(self.svm_grid.best_params_))
        print("Best Test Score: \n{}\n".format(self.svm_grid.best_score_))

        self.optimised_model = self.svm_grid.best_estimator_
        return [self.optimised_model, self.svm_grid.best_score_]

    def fit(self, X_train, y_train):
        print('Training ', self.model_type)
        self.optimised_model.fit(X_train, y_train)
        print('Completed Training')

        return self.optimised_model

    def predict(self, X_val):
        print('Predicting unlabelled...')
        self.val_y_predicted = self.optimised_model.predict(X_val)

        return self.val_y_predicted

    def predict_labelled(self, X_train, X_test):
        print('Predicting labelled...')
        self.train_y_predicted = self.optimised_model.predict(X_train)
        self.test_y_predicted = self.optimised_model.predict(X_test)

        return self.train_y_predicted, self.test_y_predicted

    def predict_proba(self, X_val):
        print('Proba prediction...')
        self.proba = self.optimised_model.predict_proba(X_val)
        return self.proba

    def transform_without_estimating(self, X):
        Xt = []
        pipes = [self.estimator]

    def confusion_matrix(self, y_test, y_train, class_estim):
        results_test = confusion_matrix(y_test, self.test_y_predicted, labels=class_estim)
        results_train = confusion_matrix(y_train, self.train_y_predicted, labels=class_estim)
        return results_test, results_train

    def precision_score_model(self, y_train, y_test):
        score_train = precision_score(y_true=y_train, y_pred=self.train_y_predicted)
        score_test = precision_score(y_true=y_test, y_pred=self.test_y_predicted)
        return score_train, score_test


class RfModel(BaseModel):
    model_type = 'Random_Forest'
    model_category = 'Classification'

    def __init__(self):
        self.proba = None
        self.optimised_model = None
        self.test_y_predicted = None
        self.val_y_predicted = None
        self.train_y_predicted = None
        self.classifier = None
        self.optimised_rf = None
        self.rf_grid = None
        self.deopt_classifier = None

    def gridsearch(self, X_train, y_train, c_weight, catboost_weight: Optional[None], splits: int = 5,
                   scoring_type: str = 'precision',
                   kfold_shuffle: bool = True,
                   parameters=None):
        print('Gridsearching RFModel...')
        self.deopt_classifier = RandomForestClassifier(class_weight=c_weight)
        self.stratifiedkfold = StratifiedKFold(n_splits=splits, shuffle=kfold_shuffle, random_state=42)


        self.rf_grid = GridSearchCV(self.deopt_classifier, param_grid= parameters, cv=self.stratifiedkfold, refit=True,
                                    scoring=scoring_type,
                                    verbose=10, n_jobs=-1)

        self.rf_grid.fit(X_train, y_train)
        print("Best Estimator: \n{}\n".format(self.rf_grid.best_estimator_))
        print("Best Parameters: \n{}\n".format(self.rf_grid.best_params_))
        print("Best Test Score: \n{}\n".format(self.rf_grid.best_score_))

        self.optimised_model = self.rf_grid.best_estimator_
        return [self.optimised_model, self.rf_grid.best_score_]

    def fit(self, X_train, y_train):
        print('Training ', self.model_type)
        self.optimised_model.fit(X_train, y_train)
        print('Completed Training')

        return self.optimised_model

    def predict(self, X_val):
        print('Predicting unlabelled...')
        self.val_y_predicted = self.optimised_model.predict(X_val)

        return self.val_y_predicted

    def predict_labelled(self, X_train, X_test):
        print('Predicting labelled...')
        self.train_y_predicted = self.optimised_model.predict(X_train)
        self.test_y_predicted = self.optimised_model.predict(X_test)

        return self.train_y_predicted, self.test_y_predicted

    def predict_proba(self, X_val):
        print('Proba prediction...')
        self.proba = self.optimised_model.predict_proba(X_val)
        return self.proba

    def confusion_matrix(self, y_test, y_train, class_estim):
        results_test = confusion_matrix(y_test, self.test_y_predicted, labels=class_estim)
        results_train = confusion_matrix(y_train, self.train_y_predicted, labels=class_estim)
        return results_test, results_train

    def precision_score_model(self, y_train, y_test):
        score_train = precision_score(y_true=y_train, y_pred=self.train_y_predicted)
        score_test = precision_score(y_true=y_test, y_pred=self.test_y_predicted)
        return score_train, score_test


class CBModel(BaseModel):
    model_type = 'CatBoostClass'
    model_category = 'Classification'

    def __init__(self):
        self.proba = None
        self.test_y_predicted = None
        self.val_y_predicted = None
        self.train_y_predicted = None
        self.optimised_model = None
        self.classifier = None
        self.paramgrid = None
        self.deopt_classifier = None
        self.optimised_cbmodel = None

        self.eval_metric_map = {'accuracy': 'Accuracy', 'balanced_accurary': 'BalancedAccuracy', 'f1': 'F1',
                                'recall': 'Recall', 'precision': 'Precision'}

    def gridsearch(self, X_train, y_train, c_weight, catboost_weight: Optional[None], splits: int = 5,
                   scoring_type: str = 'Precision',
                   kfold_shuffle: bool = True,
                   parameters = None):
        print('GridSearching CatBoost...')

        if scoring_type in self.eval_metric_map:
            scoring = self.eval_metric_map[scoring_type]
        else:
            print('Scoring method for CatBoost not found in mapping. Defaulting to "Cross Entropy"')
            scoring = 'CrossEntropy'

        weights = compute_class_weight(class_weight='balanced', classes=catboost_weight, y=y_train)
        class_weights = dict(zip(catboost_weight, weights))

        self.deopt_classifier = cb.CatBoostClassifier(loss_function='Logloss', eval_metric=scoring,
                                                      auto_class_weights="Balanced", early_stopping_rounds=42,
                                                      # class_weights=class_weights,

                                                      random_seed=42)
        self.stratifiedkfold = StratifiedKFold(n_splits=splits, shuffle=kfold_shuffle, random_state=42)


        self.cb_grid = self.deopt_classifier.grid_search(parameters, X_train, y_train, cv=self.stratifiedkfold,
                                                         calc_cv_statistics=True, search_by_train_test_split=True,
                                                         # Set this parameter to true or it does not work "Classlables in
                                                         # dataprocessing option do not match training
                                                         refit=True,
                                                         shuffle=False, verbose=False, plot=False, log_cout=sys.stdout,
                                                         log_cerr=sys.stderr)

        # self.cb_grid = GridSearchCV(self.deopt_classifier, self.paramgrid, cv=self.stratifiedkfold, refit=True,
        # verbose=10, n_jobs=-1)
        # self.cb_grid.fit(X_train, y_train)
        print("Best Estimator: \n{}\n".format(self.cb_grid['params']))
        print("Best Parameters: \n{}\n".format(self.cb_grid['cv_results']))
        # print("Best Test Score: \n{}\n".format(self.cb_grid.best_score_))
        # I believe after the gridsearch is completed, the deoptimised classifier is set with the best params
        self.optimised_model = self.deopt_classifier
        self.best_score = max(self.cb_grid['cv_results']['test-' + str(scoring) + '-mean'])
        return (self.cb_grid['params'], self.best_score)

    def fit(self, X_train, y_train):
        print('Training ', self.model_type)
        self.optimised_model.fit(X_train, y_train)
        print('Completed Training')

        return self.optimised_model

    def predict(self, X_val):
        print('Predicting unlabelled...')
        self.val_y_predicted = self.optimised_model.predict(X_val)

        return self.val_y_predicted

    def predict_labelled(self, X_train, X_test):
        print('Predicting labelled...')
        self.train_y_predicted = self.optimised_model.predict(X_train)
        self.test_y_predicted = self.optimised_model.predict(X_test)

        return self.train_y_predicted, self.test_y_predicted

    def predict_proba(self, X_val):
        print('Proba prediction...')
        self.proba = self.optimised_model.predict_proba(X_val)
        return self.proba

    def confusion_matrix(self, y_test, y_train, class_estim):
        results_test = confusion_matrix(y_test, self.test_y_predicted, labels=class_estim)
        results_train = confusion_matrix(y_train, self.train_y_predicted, labels=class_estim)
        return results_test, results_train

    def precision_score_model(self, y_train, y_test):
        score_train = precision_score(y_true=y_train, y_pred=self.train_y_predicted)
        score_test = precision_score(y_true=y_test, y_pred=self.test_y_predicted)
        return score_train, score_test


####Regregression Models

class SVR_Model(BaseModel):
    model_type = 'SVR'
    model_category = 'Regression'

    def __init__(self):
        self.kfold = None
        self.paramgrid = None
        self.proba = None
        self.optimised_model = None
        self.val_y_predicted = None
        self.test_y_predicted = None
        self.stratifiedkfold = None
        self.deopt_classifier = None
        self.train_y_predicted = None

    def gridsearch(self, X_train, y_train, params: dict=None, splits: int = 5, kfold_shuffle: bool = True,
                   scoring_type: str = 'explained_variance', verbose: int = 0):
        print("GridSearching SVR...")
        self.deopt_classifier = SVR()
        self.kfold = KFold(n_splits=splits, shuffle=kfold_shuffle, random_state=42)
        # TODO Better define params

        # TODO Create kFold&Scoring PARAM choose
        self.svm_grid = GridSearchCV(self.deopt_classifier,param_grid=params , cv=self.kfold, refit=True,
                                     n_jobs=-1,
                                     verbose=verbose,
                                     scoring=scoring_type)
        self.svm_grid.fit(X_train, y_train)

        # print("Best Estimator: \n{}\n".format(self.svm_grid.best_estimator_))
        # print("Best Parameters: \n{}\n".format(self.svm_grid.best_params_))
        # print("Best Test Score: \n{}\n".format(self.svm_grid.best_score_))

        self.optimised_model = self.svm_grid.best_estimator_
        return [self.optimised_model, self.svm_grid.best_score_]

    def fit(self, X_train, y_train):
        print('Training ', self.model_type)
        self.optimised_model.fit(X_train, y_train)
        print('Completed Training')

        return self.optimised_model

    def predict(self, X_val):
        print('Predicting unlabelled...')
        self.val_y_predicted = self.optimised_model.predict(X_val)

        return self.val_y_predicted

    def predict_labelled(self, X_train, X_test):
        print('Predicting labelled...')
        self.train_y_predicted = self.optimised_model.predict(X_train)
        self.test_y_predicted = self.optimised_model.predict(X_test)

        return self.train_y_predicted, self.test_y_predicted

    def predict_proba(self, X_val):
        print('Proba prediction...')
        self.proba = self.optimised_model.predict_proba(X_val)
        return self.proba

    def predict_actual_graph(self, y_actual_train, y_actual_test, score_query, save_path: Optional[str], plot: bool=False):

        score_train = score_query(y_actual_train, self.train_y_predicted)
        score_test = score_query(y_actual_test, self.test_y_predicted)
        plt.figure(figsize=(10, 10))
        plt.scatter(y_actual_train, self.train_y_predicted, c='crimson')
        plt.yscale('log')
        plt.xscale('log')

        p1 = max(max(self.train_y_predicted), max(y_actual_train))
        p2 = min(min(self.train_y_predicted), min(y_actual_train))
        plt.plot([p1, p2], [p1, p2], 'b-')
        plt.xlabel('True Values', fontsize=15)
        plt.ylabel('Predictions', fontsize=15)
        plt.title(str(SVR_Model.model_type) + ': ' + str(score_train))
        plt.axis('equal')

        if plot == True:
            plt.show()
        elif plot == False:
            plot_name = str(SVR_Model.model_type) + '_actual_vs_prediction_train.jpg'
            plot_name = os.path.join(save_path, plot_name)
            plt.tight_layout()
            plt.savefig(plot_name, dpi = 400)
        print(plt.get_backend())

        # close any existing plots
        plt.close("all")
        plt.figure(figsize=(10, 10))
        plt.scatter(y_actual_test, self.test_y_predicted, c='crimson')
        plt.yscale('log')
        plt.xscale('log')

        p1 = max(max(self.test_y_predicted), max(y_actual_test))
        p2 = min(min(self.test_y_predicted), min(y_actual_test))
        plt.plot([p1, p2], [p1, p2], 'b-')
        plt.xlabel('True Values', fontsize=15)
        plt.ylabel('Predictions', fontsize=15)
        plt.title(str(SVR_Model.model_type) + ': ' + str(score_test))
        plt.axis('equal')

        if plot == True:
            plt.show()
        elif plot == False:
            plot_name = str(SVR_Model.model_type) + '_actual_vs_prediction_test.jpg'
            plot_name = os.path.join(save_path, plot_name)
            plt.tight_layout()
            plt.savefig(plot_name, dpi=400)


class RandomForestEnsemble(BaseModel):
    model_type = 'RFE_Regressor'
    model_category = 'Regression'

    def __init__(self):
        self.kfold = None
        self.paramgrid = None
        self.proba = None
        self.optimised_model = None
        self.val_y_predicted = None
        self.test_y_predicted = None
        self.stratifiedkfold = None
        self.deopt_classifier = None
        self.train_y_predicted = None

    def gridsearch(self, X_train, y_train, params: dict=None, splits: int = 5, kfold_shuffle: bool = True,
                   scoring_type: str = 'explained_variance', verbose: int = 0):
        print("GridSearching RandomForestRegressor...")
        self.deopt_classifier = RandomForestRegressor(random_state=42)
        self.kfold = KFold(n_splits=splits, shuffle=kfold_shuffle, random_state=42)
        # TODO Better define params

        # The minimum number of samples required to split an internal node:
        # If int, then consider min_samples_split as the minimum number.
        # If float, then min_samples_split is a percentage and ceil(min_samples_split * n_samples)
        # are the minimum number of samples for each split.
        # TODO Create kFold&Scoring PARAM choose
        self.rf_grid = GridSearchCV(self.deopt_classifier, param_grid=params, cv=self.kfold, refit=True,
                                    n_jobs=-1,
                                    verbose=verbose,
                                    scoring=scoring_type)
        self.rf_grid.fit(X_train, y_train)

        print("Best Estimator: \n{}\n".format(self.rf_grid.best_estimator_))
        print("Best Parameters: \n{}\n".format(self.rf_grid.best_params_))
        print("Best Test Score: \n{}\n".format(self.rf_grid.best_score_))

        self.optimised_model = self.rf_grid.best_estimator_
        return [self.optimised_model, self.rf_grid.best_score_]

    def fit(self, X_train, y_train):
        print('Training ', self.model_type)
        self.optimised_model.fit(X_train, y_train)
        print('Completed Training')

        return self.optimised_model

    def predict(self, X_val):
        print('Predicting unlabelled...')
        self.val_y_predicted = self.optimised_model.predict(X_val)

        return self.val_y_predicted

    def predict_labelled(self, X_train, X_test):
        print('Predicting labelled...')
        self.train_y_predicted = self.optimised_model.predict(X_train)
        self.test_y_predicted = self.optimised_model.predict(X_test)

        return self.train_y_predicted, self.test_y_predicted

    def predict_actual_graph(self, y_actual_train, y_actual_test, score_query, save_path: Optional[str], plot: bool=False):

        score_train = score_query(y_actual_train, self.train_y_predicted)
        score_test = score_query(y_actual_test, self.test_y_predicted)
        plt.figure(figsize=(10, 10))
        plt.scatter(y_actual_train, self.train_y_predicted, c='crimson')
        plt.yscale('log')
        plt.xscale('log')

        p1 = max(max(self.train_y_predicted), max(y_actual_train))
        p2 = min(min(self.train_y_predicted), min(y_actual_train))
        plt.plot([p1, p2], [p1, p2], 'b-')
        plt.xlabel('True Values', fontsize=15)
        plt.ylabel('Predictions', fontsize=15)
        plt.title(str(RandomForestEnsemble.model_type) + ': ' + str(score_train))
        plt.axis('equal')

        if plot == True:
            plt.show()
        elif plot == False:
            plot_name = str(RandomForestEnsemble.model_type) + '_actual_vs_prediction_train.jpg'
            plot_name = os.path.join(save_path, plot_name)
            plt.tight_layout()
            plt.savefig(plot_name, dpi = 400)
        print(plt.get_backend())

        # close any existing plots
        plt.close("all")
        plt.figure(figsize=(10, 10))
        plt.scatter(y_actual_test, self.test_y_predicted, c='crimson')
        plt.yscale('log')
        plt.xscale('log')

        p1 = max(max(self.test_y_predicted), max(y_actual_test))
        p2 = min(min(self.test_y_predicted), min(y_actual_test))
        plt.plot([p1, p2], [p1, p2], 'b-')
        plt.xlabel('True Values', fontsize=15)
        plt.ylabel('Predictions', fontsize=15)
        plt.title(str(RandomForestEnsemble.model_type) + ': ' + str(score_test))
        plt.axis('equal')

        if plot == True:
            plt.show()
        elif plot == False:
            plot_name = str(RandomForestEnsemble.model_type) + '_actual_vs_prediction_test.jpg'
            plot_name = os.path.join(save_path, plot_name)
            plt.tight_layout()
            plt.savefig(plot_name, dpi=400)


class CatBoostReg(BaseModel):
    model_type = 'CatBoostReg'
    model_category = 'Regression'

    def __init__(self):
        self.kfold = None
        self.paramgrid = None
        self.proba = None
        self.optimised_model = None
        self.val_y_predicted = None
        self.test_y_predicted = None
        self.stratifiedkfold = None
        self.deopt_classifier = None
        self.train_y_predicted = None

        self.eval_metric_map = {'neg_root_mean_squared_error': 'RMSE',
                                'neg_mean_squared_log_error': 'MSLE',
                                'neg_mean_absolute_error': 'MAE',
                                'r2': 'R2',
                                'neg_mean_absolute_percentage_error': 'MAPE',
                                'neg_median_absolute_error': 'MedianAbsoluteError'}

    def gridsearch(self, X_train, y_train, params: dict=None, splits: int = 5, kfold_shuffle: bool = True,
                   scoring_type: str = 'Poisson', verbose: int = 0):
        print('Gridsearching CatBoost Regressor...')

        if scoring_type in self.eval_metric_map:
            scoring = self.eval_metric_map[scoring_type]
        else:
            print('Scoring method for CatBoost not found in mapping. Defaulting to "Poisson"')
            scoring = 'Poisson'

        self.deopt_classifier = cb.CatBoostRegressor(loss_function='RMSE', random_seed=42, eval_metric=scoring,
                                                     early_stopping_rounds=42)
        # https://towardsdatascience.com/5-cute-features-of-catboost-61532c260f69


        self.kfold = KFold(n_splits=splits, shuffle=kfold_shuffle, random_state=42)

        self.cb_grid = self.deopt_classifier.grid_search(params, X_train, y_train, cv=self.kfold,
                                                         calc_cv_statistics=True, refit=True, verbose=verbose, shuffle=False,
                                                         log_cout=sys.stdout,
                                                         log_cerr=sys.stderr
                                                         )
        print("Best Estimator: \n{}\n".format(self.cb_grid['params']))
        self.optimised_model = self.deopt_classifier

        self.best_score = max(self.cb_grid['cv_results']['test-' + str(scoring) + '-mean'])
        return (self.cb_grid['params'], self.best_score)

    def fit(self, X_train, y_train):
        print('Training ', self.model_type)
        self.optimised_model.fit(X_train, y_train)
        print('Completed Training')

        return self.optimised_model

    def predict(self, X_val):
        print('Predicting unlabelled...')
        self.val_y_predicted = self.optimised_model.predict(X_val)

        return self.val_y_predicted

    def predict_labelled(self, X_train, X_test):
        print('Predicting labelled...')
        self.train_y_predicted = self.optimised_model.predict(X_train)
        self.test_y_predicted = self.optimised_model.predict(X_test)

        return self.train_y_predicted, self.test_y_predicted

    def predict_actual_graph(self, y_actual_train, y_actual_test, score_query, save_path: Optional[str], plot: bool=False):

        score_train = score_query(y_actual_train, self.train_y_predicted)
        score_test = score_query(y_actual_test, self.test_y_predicted)
        plt.figure(figsize=(10, 10))
        plt.scatter(y_actual_train, self.train_y_predicted, c='crimson')
        plt.yscale('log')
        plt.xscale('log')

        p1 = max(max(self.train_y_predicted), max(y_actual_train))
        p2 = min(min(self.train_y_predicted), min(y_actual_train))
        plt.plot([p1, p2], [p1, p2], 'b-')
        plt.xlabel('True Values', fontsize=15)
        plt.ylabel('Predictions', fontsize=15)
        plt.title(str(CatBoostReg.model_type) + ': ' + str(score_train))
        plt.axis('equal')

        if plot == True:
            plt.show()
        elif plot == False:
            plot_name = str(CatBoostReg.model_type) + '_actual_vs_prediction_train.jpg'
            plot_name = os.path.join(save_path, plot_name)
            plt.tight_layout()
            plt.savefig(plot_name, dpi = 400)
        print(plt.get_backend())

        # close any existing plots
        plt.close("all")
        plt.figure(figsize=(10, 10))
        plt.scatter(y_actual_test, self.test_y_predicted, c='crimson')
        plt.yscale('log')
        plt.xscale('log')

        p1 = max(max(self.test_y_predicted), max(y_actual_test))
        p2 = min(min(self.test_y_predicted), min(y_actual_test))
        plt.plot([p1, p2], [p1, p2], 'b-')
        plt.xlabel('True Values', fontsize=15)
        plt.ylabel('Predictions', fontsize=15)
        plt.title(str(CatBoostReg.model_type) + ': ' + str(score_test))
        plt.axis('equal')

        if plot == True:
            plt.show()
        elif plot == False:
            plot_name = str(CatBoostReg.model_type) + '_actual_vs_prediction_test.jpg'
            plot_name = os.path.join(save_path, plot_name)
            plt.tight_layout()
            plt.savefig(plot_name, dpi=400)

class SVRLinear(BaseModel):

    model_type = 'SVR_Linear'
    model_category = 'Regression'

    def __init__(self):
        self.kfold = None
        self.paramgrid = None
        self.proba = None
        self.optimised_model = None
        self.val_y_predicted = None
        self.test_y_predicted = None
        self.stratifiedkfold = None
        self.deopt_classifier = None
        self.train_y_predicted = None
    def gridsearch(self, X_train, y_train, params: dict=None, splits: int = 5, kfold_shuffle: bool = True,
                   scoring_type: str = 'explained_variance', verbose: int=0):
        print("GridSearching SVRLinear...")
        self.deopt_classifier = LinearSVR(random_state=42)
        self.kfold = KFold(n_splits=splits, shuffle=kfold_shuffle, random_state=42)
        # TODO Better define params

        # TODO Create kFold&Scoring PARAM choose
        self.svm_grid = GridSearchCV(self.deopt_classifier,param_grid=params , cv=self.kfold, refit=True,
                                     n_jobs=-1,
                                     verbose=verbose,
                                     scoring=scoring_type)
        self.svm_grid.fit(X_train, y_train)

        # print("Best Estimator: \n{}\n".format(self.svm_grid.best_estimator_))
        # print("Best Parameters: \n{}\n".format(self.svm_grid.best_params_))
        # print("Best Test Score: \n{}\n".format(self.svm_grid.best_score_))
        self.optimised_model = self.svm_grid.best_estimator_
        return [self.optimised_model, self.svm_grid.best_score_]

    def fit(self, X_train, y_train):
        print('Training ', self.model_type)
        self.optimised_model.fit(X_train, y_train)
        print('Completed Training')

        return self.optimised_model
    def predict(self, X_val):
        print('Predicting unlabelled...')
        self.val_y_predicted = self.optimised_model.predict(X_val)

        return self.val_y_predicted

    def predict_labelled(self, X_train, X_test):
        print('Predicting labelled...')
        self.train_y_predicted = self.optimised_model.predict(X_train)
        self.test_y_predicted = self.optimised_model.predict(X_test)

        return self.train_y_predicted, self.test_y_predicted

    def predict_proba(self, X_val):
        print('Proba prediction...')
        self.proba = self.optimised_model.predict_proba(X_val)
        return self.proba
    def predict_actual_graph(self, y_actual_train, y_actual_test, score_query, save_path: Optional[str], plot: bool=False):

        score_train = score_query(y_actual_train, self.train_y_predicted)
        score_test = score_query(y_actual_test, self.test_y_predicted)
        plt.figure(figsize=(10, 10))
        plt.scatter(y_actual_train, self.train_y_predicted, c='crimson')
        plt.yscale('log')
        plt.xscale('log')

        p1 = max(max(self.train_y_predicted), max(y_actual_train))
        p2 = min(min(self.train_y_predicted), min(y_actual_train))
        plt.plot([p1, p2], [p1, p2], 'b-')
        plt.xlabel('True Values', fontsize=15)
        plt.ylabel('Predictions', fontsize=15)
        plt.title(str(SVRLinear.model_type) + ': ' + str(score_train))
        plt.axis('equal')

        if plot == True:
            plt.show()
        elif plot == False:
            plot_name = str(SVRLinear.model_type) + '_actual_vs_prediction_train.jpg'
            plot_name = os.path.join(save_path, plot_name)
            plt.tight_layout()
            plt.savefig(plot_name, dpi = 400)
        print(plt.get_backend())

        # close any existing plots
        plt.close("all")
        plt.figure(figsize=(10, 10))
        plt.scatter(y_actual_test, self.test_y_predicted, c='crimson')
        plt.yscale('log')
        plt.xscale('log')

        p1 = max(max(self.test_y_predicted), max(y_actual_test))
        p2 = min(min(self.test_y_predicted), min(y_actual_test))
        plt.plot([p1, p2], [p1, p2], 'b-')
        plt.xlabel('True Values', fontsize=15)
        plt.ylabel('Predictions', fontsize=15)
        plt.title(str(SVRLinear.model_type) + ': ' + str(score_test))
        plt.axis('equal')

        if plot == True:
            plt.show()
        elif plot == False:
            plot_name = str(SVRLinear.model_type) + '_actual_vs_prediction_test.jpg'
            plot_name = os.path.join(save_path, plot_name)
            plt.tight_layout()
            plt.savefig(plot_name, dpi=400)


class Multi_SVR(BaseModel):

    def __init__(self):
        self.kfold = None
        self.paramgrid = None
        self.proba = None
        self.optimised_model = None
        self.val_y_predicted = None
        self.test_y_predicted = None
        self.stratifiedkfold = None
        self.deopt_classifier = None
        self.train_y_predicted = None


    def gridsearch(self, X_train, y_train, params: dict=None, splits: int = 5, kfold_shuffle: bool = True,
                   scoring_type: str = 'explained_variance'):
        print("GridSearching SVR...")
        self.deopt_classifier = MSVR(kernel='rbf', gamma=0.1, epsilon=0.001)
        self.kfold = KFold(n_splits=splits, shuffle=kfold_shuffle, random_state=42)
        # TODO Better define params

        # TODO Create kFold&Scoring PARAM choose
        self.svm_grid = GridSearchCV(self.deopt_classifier,param_grid=params , cv=self.kfold, refit=True,
                                     n_jobs=-1,
                                     verbose=10,
                                     scoring=scoring_type)
        self.svm_grid.fit(X_train, y_train)

        # print("Best Estimator: \n{}\n".format(self.svm_grid.best_estimator_))
        # print("Best Parameters: \n{}\n".format(self.svm_grid.best_params_))
        # print("Best Test Score: \n{}\n".format(self.svm_grid.best_score_))

        self.optimised_model = self.svm_grid.best_estimator_
        return [self.optimised_model, self.svm_grid.best_score_]
    def fit(self, X_train, y_train):
        print('Training ', self.model_type)
        self.optimised_model.fit(X_train, y_train)
        print('Completed Training')

        return self.optimised_model

    def predict(self, X_val):
        print('Predicting unlabelled...')
        self.val_y_predicted = self.optimised_model.predict(X_val)

        return self.val_y_predicted

    def predict_labelled(self, X_train, X_test):
        print('Predicting labelled...')
        self.train_y_predicted = self.optimised_model.predict(X_train)
        self.test_y_predicted = self.optimised_model.predict(X_test)

        return self.train_y_predicted, self.test_y_predicted

    def predict_proba(self, X_val):
        print('Proba prediction...')
        self.proba = self.optimised_model.predict_proba(X_val)
        return self.proba

    def predict_actual_graph(self, y_actual_train, y_actual_test, score_query, save_path: Optional[str], plot: bool=False):

        score_train = score_query(y_actual_train, self.train_y_predicted)
        score_test = score_query(y_actual_test, self.test_y_predicted)
        plt.figure(figsize=(10, 10))
        plt.scatter(y_actual_train, self.train_y_predicted, c='crimson')
        plt.yscale('log')
        plt.xscale('log')

        p1 = max(max(self.train_y_predicted), max(y_actual_train))
        p2 = min(min(self.train_y_predicted), min(y_actual_train))
        plt.plot([p1, p2], [p1, p2], 'b-')
        plt.xlabel('True Values', fontsize=15)
        plt.ylabel('Predictions', fontsize=15)
        plt.title(str(RandomForestEnsemble.model_type) + ': ' + str(score_train))
        plt.axis('equal')

        if plot == True:
            plt.show()
        elif plot == False:
            plot_name = str(SVRLinear.model_type) + '_actual_vs_prediction_train.jpg'
            plot_name = os.path.join(save_path, plot_name)
            plt.tight_layout()
            plt.savefig(plot_name, dpi = 400)
        print(plt.get_backend())

        # close any existing plots
        plt.close("all")
        plt.figure(figsize=(10, 10))
        plt.scatter(y_actual_test, self.test_y_predicted, c='crimson')
        plt.yscale('log')
        plt.xscale('log')

        p1 = max(max(self.test_y_predicted), max(y_actual_test))
        p2 = min(min(self.test_y_predicted), min(y_actual_test))
        plt.plot([p1, p2], [p1, p2], 'b-')
        plt.xlabel('True Values', fontsize=15)
        plt.ylabel('Predictions', fontsize=15)
        plt.title(str(RandomForestEnsemble.model_type) + ': ' + str(score_test))
        plt.axis('equal')

        if plot == True:
            plt.show()
        elif plot == False:
            plot_name = str(SVRLinear.model_type) + '_actual_vs_prediction_test.jpg'
            plot_name = os.path.join(save_path, plot_name)
            plt.tight_layout()
            plt.savefig(plot_name, dpi=400)