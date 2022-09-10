import os
import sys
from typing import Optional

import catboost as cb
import numpy as np
import pandas as pd
import shap
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import confusion_matrix, precision_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV, KFold, RepeatedKFold, cross_val_score, \
    RandomizedSearchCV
from sklearn.svm import SVC, SVR, LinearSVR
from sklearn.utils import compute_class_weight

import ToolsActiveLearning
from BaseModel import BaseModel
from MSVR import MSVR
from shapash.explainer.smart_explainer import SmartExplainer

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
                   parameters=None):
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

        self.rf_grid = GridSearchCV(self.deopt_classifier, param_grid=parameters, cv=self.stratifiedkfold, refit=True,
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
                   parameters=None):
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
        self.cv_results = None
        self.kfold = None
        self.paramgrid = None
        self.proba = None
        self.optimised_model = None
        self.val_y_predicted = None
        self.test_y_predicted = None
        self.stratifiedkfold = None
        self.deopt_classifier = None
        self.train_y_predicted = None

    def gridsearch(self, X_train: np.ndarray = None, y_train: np.ndarray = None, initialisation: str = None,
                   params: dict = None, splits: int = 5,
                   kfold_shuffle: int = 1,
                   scoring_type: str = 'explained_variance', verbose: int = 0):
        print("GridSearching SVR...")
        self.deopt_classifier = SVR()
        self.kfold = RepeatedKFold(n_splits=splits, n_repeats=kfold_shuffle, random_state=42)
        # TODO Better define params

        # TODO Create kFold&Scoring PARAM choose
        if initialisation == "gridsearch":
            self.svm_grid = GridSearchCV(self.deopt_classifier, param_grid=params, cv=self.kfold, refit=True,
                                         n_jobs=-1,
                                         verbose=verbose,
                                         scoring=scoring_type)
            self.svm_grid.fit(X_train, y_train)
        elif initialisation == "randomized":
            self.svm_grid = RandomizedSearchCV(self.deopt_classifier, param_distributions=params, n_iter=60, refit=True,
                                               n_jobs=-1,
                                               verbose=verbose,
                                               cv=self.kfold,
                                               scoring=scoring_type)
            self.svm_grid.fit(X_train, y_train)

        # print("Best Estimator: \n{}\n".format(self.svm_grid.best_estimator_))
        # print("Best Parameters: \n{}\n".format(self.svm_grid.best_params_))
        # print("Best Test Score: \n{}\n".format(self.svm_grid.best_score_))
        self.cv_results = pd.DataFrame(self.svm_grid.cv_results_)
        self.optimised_model = self.svm_grid.best_estimator_
        return [self.optimised_model, self.svm_grid.best_score_]

    def default_model(self, X_train: np.ndarray = None, y_train: np.ndarray = None, params: dict = None, splits: int = 5,
                  kfold_shuffle: int = 1, scoring_type: str = 'r2', verbose: int = 0
                  ):
        self.kfold = RepeatedKFold(n_splits=splits, n_repeats=kfold_shuffle, random_state=42)
        self.optimised_model = SVR()
        self.cv_score = cross_val_score(self.optimised_model, X_train, y_train, scoring=scoring_type, cv=self.kfold,
                                        n_jobs=-1)
        self.best_score = np.mean(self.cv_score)
        print("SVR: ", self.best_score)
        return [self.optimised_model, self.best_score]

    def optimised(self, X_train: np.ndarray = None, y_train: np.ndarray = None, params: dict = None, splits: int = 5,
                  kfold_shuffle: int = 1, scoring_type: str = 'r2', verbose: int = 0
                  ):

        self.kfold = RepeatedKFold(n_splits=splits, n_repeats=kfold_shuffle, random_state=42)
        self.optimised_model = SVR()
        self.optimised_model.set_params(**params)

        self.cv_score = cross_val_score(self.optimised_model, X_train, y_train, scoring=scoring_type, cv=self.kfold,
                                        n_jobs=-1)
        self.best_score = np.mean(self.cv_score)
        print("SVR: ", self.best_score)
        return [self.optimised_model, self.best_score]

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

    def predict_actual_graph(self, y_actual_train, y_actual_test, score_query, save_path: Optional[str],
                             plot: bool = False):

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
            plt.savefig(plot_name, dpi=400)
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
        plt.title(str(SVR_Model.model_type) + ': ' + str(np.round(score_test,2)))
        plt.axis('equal')

        if plot == True:
            plt.show()
        elif plot == False:
            plot_name = str(SVR_Model.model_type) + '_actual_vs_prediction_test.jpg'
            plot_name = os.path.join(save_path, plot_name)
            plt.tight_layout()
            plt.savefig(plot_name, dpi=400)


    def shap_analysis_model(self,X_test,X, features, y_test, save_path):
        #Fit the Explainer
        #temp_df = pd.DataFrame(X_test, columns=[features])
        explainer = shap.Explainer(self.optimised_model.predict, X_test, feature_names=features)
        #Calculate the SHAP values
        shap_values = explainer(X_test)
            #.data is the copy of the input data
            #.base_values is the expected value of the target
            #.values are the SHAP values for each example
        plt.rcParams['axes.xmargin'] = 0
        plt.close("all")
        shap.plots.bar(shap_values)
        plt.close("all")
        #Maybe cluster with full dataset? - Based on XGBoost
        clustering = shap.utils.hclust(X_test, y_test)
        shap.plots.bar(shap_values, clustering=clustering, clustering_cutoff=0.9)
        plt.close("all")
        #fig = shap.plots.bar(shap_values, clustering=clustering, clustering_cutoff=0.9, show=False)
        #fig_summary = str(SVR_Model.model_type) + "_all_predictions_test_Regression.jpg"
        #fig_summary = os.path.join(save_path, fig_summary)
        #fig.savefig(fig_summary, bbox_inches='tight')
        ToolsActiveLearning.print_feature_importance_shap_values(shap_values, features)




class RandomForestEnsemble(BaseModel):
    model_type = 'RFE_Regressor'
    model_category = 'Regression'

    def __init__(self):
        self.cv_results = None
        self.kfold = None
        self.paramgrid = None
        self.proba = None
        self.optimised_model = None
        self.val_y_predicted = None
        self.test_y_predicted = None
        self.stratifiedkfold = None
        self.deopt_classifier = None
        self.train_y_predicted = None

    def gridsearch(self, X_train, y_train, initialisation: str, params: dict = None, splits: int = 5,
                   kfold_shuffle: int = 1,
                   scoring_type: str = 'explained_variance', verbose: int = 0):
        print("GridSearching RandomForestRegressor...")
        self.deopt_classifier = RandomForestRegressor(random_state=42)
        self.kfold = RepeatedKFold(n_splits=splits, n_repeats=kfold_shuffle, random_state=42)
        # TODO Better define params

        # The minimum number of samples required to split an internal node:
        # If int, then consider min_samples_split as the minimum number.
        # If float, then min_samples_split is a percentage and ceil(min_samples_split * n_samples)
        # are the minimum number of samples for each split.
        # TODO Create kFold&Scoring PARAM choose
        if initialisation == 'gridsearch':
            self.rf_grid = GridSearchCV(self.deopt_classifier, param_grid=params, cv=self.kfold, refit=True,
                                        n_jobs=-1,
                                        verbose=verbose,
                                        scoring=scoring_type)
            self.rf_grid.fit(X_train, y_train)
        elif initialisation == "randomized":
            self.rf_grid = RandomizedSearchCV(self.deopt_classifier, param_distributions=params, n_iter=60, refit=True,
                                              n_jobs=-1,
                                              verbose=verbose,
                                              cv=self.kfold,
                                              scoring=scoring_type)
            self.rf_grid.fit(X_train, y_train)

        print("Best Estimator: \n{}\n".format(self.rf_grid.best_estimator_))
        print("Best Parameters: \n{}\n".format(self.rf_grid.best_params_))
        print("Best Test Score: \n{}\n".format(self.rf_grid.best_score_))
        self.cv_results = pd.DataFrame(self.rf_grid.cv_results_)
        self.optimised_model = self.rf_grid.best_estimator_
        return [self.optimised_model, self.rf_grid.best_score_]

    def default_model(self, X_train: np.ndarray = None, y_train: np.ndarray = None, params: dict = None, splits: int = 5,
                  kfold_shuffle: int = 1, scoring_type: str = 'r2', verbose: int = 0
                  ):
        self.kfold = RepeatedKFold(n_splits=splits, n_repeats=kfold_shuffle, random_state=42)
        self.optimised_model = RandomForestRegressor(random_state=42)
        self.cv_score = cross_val_score(self.optimised_model, X_train, y_train, scoring=scoring_type, cv=self.kfold,
                                        n_jobs=-1)
        self.best_score = np.mean(self.cv_score)
        print("RFE: ", self.best_score)
        return [self.optimised_model, self.best_score]

    def optimised(self, X_train: np.ndarray = None, y_train: np.ndarray = None, params: dict = None, splits: int = 5,
                  kfold_shuffle: int = 1, scoring_type: str = 'explained_variance', verbose: int = 0
                  ):

        self.kfold = RepeatedKFold(n_splits=splits, n_repeats=kfold_shuffle, random_state=42)
        self.optimised_model = RandomForestRegressor(random_state=42)
        self.optimised_model.set_params(**params)

        self.cv_score = cross_val_score(self.optimised_model, X_train, y_train, scoring=scoring_type, cv=self.kfold,
                                        n_jobs=-1)
        self.best_score = np.mean(self.cv_score)
        print("RFE: ", self.best_score)
        return [self.optimised_model, self.best_score]

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

    def predict_actual_graph(self, y_actual_train, y_actual_test, score_query, save_path: Optional[str],
                             plot: bool = False):

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
            plt.savefig(plot_name, dpi=400)
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
        plt.title(str(RandomForestEnsemble.model_type) + ': ' + str(np.round(score_test,2)))
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
        self.cv_results = None
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

    def gridsearch(self, X_train, y_train, initialisation: str, params: dict = None, splits: int = 5,
                   kfold_shuffle: int = 1,
                   scoring_type: str = 'Poisson', verbose: int = 0):
        print('Gridsearching CatBoost Regressor...')

        if scoring_type in self.eval_metric_map:
            scoring = self.eval_metric_map[scoring_type]
        else:
            print('Scoring method for CatBoost not found in mapping. Defaulting to "RMSE"')
            scoring = 'RMSE'

        self.deopt_classifier = cb.CatBoostRegressor(loss_function='RMSE', random_seed=42, eval_metric=scoring
                                                     ,early_stopping_rounds=42
                                                     #, od_type="Iter"
                                                     #, od_wait=100
                                                     )
        # https://towardsdatascience.com/5-cute-features-of-catboost-61532c260f69

        self.kfold = RepeatedKFold(n_splits=splits, n_repeats=kfold_shuffle, random_state=42)

        self.cb_grid = self.deopt_classifier.grid_search(params, X_train, y_train, cv=self.kfold,
                                                         calc_cv_statistics=True, refit=True, verbose=verbose,
                                                         shuffle=False,
                                                         log_cout=sys.stdout,
                                                         log_cerr=sys.stderr
                                                         )
        print("Best Estimator: \n{}\n".format(self.cb_grid['params']))
        self.cv_results = pd.DataFrame(self.cb_grid['cv_results'])
        self.optimised_model = self.deopt_classifier

        self.best_score = max(self.cb_grid['cv_results']['test-' + str(scoring) + '-mean'])
        return (self.cb_grid['params'], self.best_score)

    def default_model(self, X_train: np.ndarray = None, y_train: np.ndarray = None, params: dict = None, splits: int = 5,
                  kfold_shuffle: int = 1, scoring_type: str = 'explained_variance', verbose: int = 0
                  ):
        self.kfold = RepeatedKFold(n_splits=splits, n_repeats=kfold_shuffle, random_state=42)
        self.optimised_model = cb.CatBoostRegressor(loss_function="RMSE", random_seed=42, verbose=False)
        self.cv_score = cross_val_score(self.optimised_model, X_train, y_train, scoring=scoring_type, cv=self.kfold,
                                        n_jobs=-1)
        self.best_score = np.mean(self.cv_score)
        print("CatBoost: ", self.best_score)
        return [self.optimised_model, self.best_score]

    def optimised(self, X_train: np.ndarray = None, y_train: np.ndarray = None, params: dict = None, splits: int = 5,
                  kfold_shuffle: int = 1, scoring_type: str = 'explained_variance', verbose: int = 0
                  ):
        if scoring_type in self.eval_metric_map:
            scoring = self.eval_metric_map[scoring_type]
        else:
            print('Scoring method for CatBoost not found in mapping. Defaulting to "RMSE"')
            scoring = 'RMSE'

        self.kfold = RepeatedKFold(n_splits=splits, n_repeats=kfold_shuffle, random_state=42)
        self.optimised_model = cb.CatBoostRegressor(loss_function='RMSE', random_seed=42, eval_metric=scoring
                                                    # ,early_stopping_rounds=42
                                                    , od_type="Iter"
                                                    , od_wait=100
                                                    )
        self.optimised_model.set_params(**params)

        self.cv_score = cross_val_score(self.optimised_model, X_train, y_train, scoring=scoring_type, cv=self.kfold,
                                        n_jobs=-1)
        self.best_score = np.mean(self.cv_score)
        print("CatBoost: ", self.best_score)

        return [self.optimised_model.get_params(), self.best_score]

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

    def predict_actual_graph(self, y_actual_train, y_actual_test, score_query, save_path: Optional[str],
                             plot: bool = False):

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
            plt.savefig(plot_name, dpi=400)
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
        plt.title(str(CatBoostReg.model_type) + ': ' + str(np.round(score_test, 2)))
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

    def gridsearch(self, X_train, y_train, params: dict = None, splits: int = 5, kfold_shuffle: bool = True,
                   scoring_type: str = 'explained_variance', verbose: int = 0, initialisation: str = 'geneticsearch'):
        print("GridSearching SVRLinear...")
        self.deopt_classifier = LinearSVR(random_state=42)
        self.kfold = KFold(n_splits=splits, shuffle=kfold_shuffle, random_state=42)
        # TODO Better define params

        # TODO Create kFold&Scoring PARAM choose
        if initialisation == "gridsearch":
            self.svm_grid = GridSearchCV(self.deopt_classifier, param_grid=params, cv=self.kfold, refit=True,
                                         n_jobs=-1,
                                         verbose=verbose,
                                         scoring=scoring_type)
            self.svm_grid.fit(X_train, y_train)

        # print("Best Estimator: \n{}\n".format(self.svm_grid.best_estimator_))
        print("Best Parameters: \n{}\n".format(self.svm_grid.best_params_))
        # print("Best Test Score: \n{}\n".format(self.svm_grid.best_score_))
        self.optimised_model = self.svm_grid.best_estimator_
        self.cv_results = pd.DataFrame(self.svm_grid.cv_results_)
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

    def predict_actual_graph(self, y_actual_train, y_actual_test, score_query, save_path: Optional[str],
                             plot: bool = False):

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
        plt.title(str(SVRLinear.model_type) + ': ' + str(np.round(score_train,2)))
        plt.axis('equal')

        if plot == True:
            plt.show()
        elif plot == False:
            plot_name = str(SVRLinear.model_type) + '_actual_vs_prediction_train.jpg'
            plot_name = os.path.join(save_path, plot_name)
            plt.tight_layout()
            plt.savefig(plot_name, dpi=400)
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

    def gridsearch(self, X_train: np.ndarray = None, y_train: np.ndarray = None, params: dict = None, splits: int = 5,
                   kfold_shuffle: bool = True,
                   scoring_type: str = 'explained_variance'):
        print("GridSearching SVR...")
        self.deopt_classifier = MSVR(kernel='rbf', gamma=0.1, epsilon=0.001)
        self.kfold = KFold(n_splits=splits, shuffle=kfold_shuffle, random_state=42)
        # TODO Better define params

        # TODO Create kFold&Scoring PARAM choose
        self.svm_grid = GridSearchCV(self.deopt_classifier, param_grid=params, cv=self.kfold, refit=True,
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

    def predict_actual_graph(self, y_actual_train, y_actual_test, score_query, save_path: Optional[str],
                             plot: bool = False):

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
        plt.title(str(RandomForestEnsemble.model_type) + ': ' + str(np.round(score_train,2)))
        plt.axis('equal')

        if plot == True:
            plt.show()
        elif plot == False:
            plot_name = str(SVRLinear.model_type) + '_actual_vs_prediction_train.jpg'
            plot_name = os.path.join(save_path, plot_name)
            plt.tight_layout()
            plt.savefig(plot_name, dpi=400)
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

# class Neural_Network(BaseModel):
#     model_type = 'NN_Reg'
#     model_category = 'Regression'
#
#     def __int__(self):
#         self.kfold = None
#         self.paramgrid = None
#         self.proba = None
#         self.optimised_model = None
#         self.val_y_predicted = None
#         self.test_y_predicted = None
#         self.stratifiedkfold = None
#         self.deopt_classifier = None
#         self.train_y_predicted = None
#
#     def make_regression_model(self, X_train, initializer='uniform', activation='relu', optimizer='SGD', loss='mse'):
#         tf.random.set_seed(42)
#         rows, cols = X_train.shape
#         training_data_samples = len(X_train)
#
#         input_layer = cols + 1
#         output_layer = 1
#         factor = 1
#         hidden_layer = (training_data_samples / factor) * (input_layer + output_layer)
#
#         model = Sequential([
#             Dense(input_layer, input_dim=input_layer, kernel_initializer=initializer, activation=activation),
#             # Input Layer
#             Dense(hidden_layer, kernel_initializer=initializer, activation=activation),  # Hidden Layer
#             Dense(output_layer, activation='linear')  # Output Layer
#         ])
#
#         model.compile(
#             loss=loss,
#             optimizer=optimizer,
#             metrics=['mse', 'mae']
#         )
#         self.model = model
#
#         return self.model
#
#     def gridsearch(self, X_train, y_train, search_init: str, params: dict = None, splits: int = 5,
#                    kfold_shuffle: int = 1,
#                    scoring_type: str = 'explained_variance', verbose: int = 0):
#
#         sgd = gradient_descent_v2.SGD()
#         reg_model = KerasRegressor(
#             build_fn=lambda: self.make_regression_model(X_train, initializer='uniform', activation='relu',
#                                                         optimizer=sgd, loss='mse'))
#         self.kfold = RepeatedKFold(n_splits=splits, n_repeats=kfold_shuffle, random_state=42)
#
#         if search_init == 'gridsearch':
#             self.nn_grid = GridSearchCV(reg_model, param_grid=params, cv=self.kfold, refit=True,
#                                         n_jobs=1,
#                                         verbose=verbose,
#                                         scoring=scoring_type)
#             self.nn_grid.fit(X_train, y_train)
#
#         self.optimised_model = self.nn_grid.best_estimator_
#         return [self.optimised_model, self.nn_grid.best_score_]
#
#     def fit(self, X_train, y_train):
#         print('Training ', self.model_type)
#         self.optimised_model.fit(X_train, y_train)
#         print('Completed Training')
#
#         return self.optimised_model
#
#     def predict(self, X_val):
#         print('Predicting unlabelled...')
#         self.val_y_predicted = self.optimised_model.predict(X_val)
#
#         return self.val_y_predicted
#
#     def predict_labelled(self, X_train, X_test):
#         print('Predicting labelled...')
#         self.train_y_predicted = self.optimised_model.predict(X_train)
#         self.test_y_predicted = self.optimised_model.predict(X_test)
#
#         return self.train_y_predicted, self.test_y_predicted
#
#     def predict_proba(self, X_val):
#         print('Proba prediction...')
#         self.proba = self.optimised_model.predict_proba(X_val)
#         return self.proba
