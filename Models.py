import sys
from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import confusion_matrix, precision_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV, KFold
from sklearn.svm import SVC, SVR
import catboost as cb
from sklearn.utils import compute_class_weight

from BaseModel import BaseModel


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
                   kfold_shuffle: bool = True):
        print("GridSearching SVM...")
        self.deopt_classifier = SVC(class_weight=c_weight, random_state=42, probability=True)
        self.stratifiedkfold = StratifiedKFold(n_splits=splits, shuffle=kfold_shuffle, random_state=42)
        # TODO Better define params
        self.paramgrid = {'C': [0.01, 0.1],  # np.logspace(-5, 2, 8),
                          'gamma': np.logspace(-5, 3, 9),
                          # Hashing out RBF provides more variation in the proba_values, however, the uniqueness 12 counts
                          'kernel': ['rbf', 'poly', 'sigmoid', 'linear'],
                          'coef0': [0, 0.001, 0.1, 1],
                          'degree': [1, 2, 3, 4]}
        # TODO Create kFold&Scoring PARAM choose
        self.svm_grid = GridSearchCV(self.deopt_classifier, self.paramgrid, cv=self.stratifiedkfold, refit=True,
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
        results_train = confusion_matrix(y_train, self.train_y_predicted, labels= class_estim)
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
                   kfold_shuffle: bool = True):
        print('Gridsearching RFModel...')
        self.deopt_classifier = RandomForestClassifier(class_weight=c_weight)
        self.stratifiedkfold = StratifiedKFold(n_splits=splits, shuffle=kfold_shuffle, random_state=42)
        self.paramgrid = {'n_estimators': [int(x) for x in np.linspace(start=200, stop=2000, num=10)],
                          'max_features': ['auto', 'sqrt'],
                           'max_depth': [int(x) for x in np.linspace(10, 110, num=11)],
                          'min_samples_split': [2, 5, 10],
                          'min_samples_leaf': [1, 2, 4],
                          'bootstrap': [True, False]}

        self.rf_grid = GridSearchCV(self.deopt_classifier, self.paramgrid, cv=self.stratifiedkfold, refit=True,
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
        results_train = confusion_matrix(y_train, self.train_y_predicted, labels= class_estim)
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
                   kfold_shuffle: bool = True):
        print('GridSearching CatBoost...')

        if scoring_type in self.eval_metric_map:
            scoring = self.eval_metric_map[scoring_type]
        else:
            print('Scoring method for CatBoost not found in mapping. Defaulting to "Cross Entropy"')
            scoring = 'CrossEntropy'

        weights = compute_class_weight(class_weight='balanced', classes=catboost_weight, y=y_train)
        class_weights = dict(zip(catboost_weight, weights))

        self.deopt_classifier = cb.CatBoostClassifier(loss_function='Logloss', eval_metric=scoring,
                                                      auto_class_weights="Balanced", early_stopping_rounds=100,
                                                      # class_weights=class_weights,
                                                      random_seed=42)
        self.stratifiedkfold = StratifiedKFold(n_splits=splits, shuffle=kfold_shuffle, random_state=42)
        self.paramgrid = {
            'depth': [3, 1, 2, 6, 4, 5, 7, 8, 9, 10]
                           ,'iterations': [250, 100, 500, 1000]
            , 'learning_rate': [0.03, 0.001, 0.01, 0.1, 0.2, 0.3]
                           ,'l2_leaf_reg': [3, 1, 5, 10, 100],
                          'border_count': [32, 5, 10, 20, 50, 100, 200]
                          }

        self.cb_grid = self.deopt_classifier.grid_search(self.paramgrid, X_train, y_train, cv=self.stratifiedkfold,
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
        self.best_score = max(self.cb_grid['cv_results']['test-'+str(scoring_type) +'-mean'])
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
        results_train = confusion_matrix(y_train, self.train_y_predicted, labels= class_estim)
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

    def gridsearch(self, X_train, y_train, splits: int = 5, kfold_shuffle: bool = True,
                   scoring_type: str = 'explained_variance'):
        print("GridSearching SVR...")
        self.deopt_classifier = SVR()
        self.kfold = KFold(n_splits=splits, shuffle=kfold_shuffle, random_state=42)
        # TODO Better define params
        self.paramgrid = {'C': [0.01, 0.1],  # np.logspace(-5, 2, 8),
                         # 'gamma': np.logspace(-3, 1, 5),
                          # Hashing out RBF provides more variation in the proba_values, however, the uniqueness 12 counts
                          #'kernel': ['rbf', 'poly', 'sigmoid', 'linear'],
                          #'coef0': [0, 0.001, 0.1, 1],
                          #'degree': [1, 3, 4],
                          #'epsilon': [0.1, 0.2, 0.3, 0.5]
                          }
        # TODO Create kFold&Scoring PARAM choose
        self.svm_grid = GridSearchCV(self.deopt_classifier, self.paramgrid, cv=self.kfold, refit=True,
                                     n_jobs=-1,
                                     verbose=10,
                                     scoring=scoring_type)
        self.svm_grid.fit(X_train, y_train)


        #print("Best Estimator: \n{}\n".format(self.svm_grid.best_estimator_))
        #print("Best Parameters: \n{}\n".format(self.svm_grid.best_params_))
        #print("Best Test Score: \n{}\n".format(self.svm_grid.best_score_))

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


class RandomForestEnsemble(BaseModel):
    model_type = 'RandomForestEnsembler Regressor'
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

    def gridsearch(self, X_train, y_train, splits: int = 5, kfold_shuffle: bool = True,
                   scoring_type: str = 'explained_variance'):
        print("GridSearching RandomForestRegressor...")
        self.deopt_classifier = RandomForestRegressor(random_state=42)
        self.kfold = KFold(n_splits=splits, shuffle=kfold_shuffle, random_state=42)
        # TODO Better define params
        self.paramgrid = {'n_estimators': [int(x) for x in np.linspace(start=200,
                                                                       stop=1000,
                                                                       num=9)],
                          #'criterion': ['squared_error', 'absolute_error', 'poisson'],
                          #'max_features': ['auto', 'sqrt', 'log2'],
                          # Hashing out RBF provides more variation in the proba_values, however, the uniqueness 12 counts
                         # 'max_depth': [int(x) for x in np.linspace(1, 110,num=12)],
                          #'bootstrap': [False, True],
                          #'min_samples_leaf': [float(x) for x in np.arange(0.1, 0.6, 0.1)]
                          }
        # The minimum number of samples required to split an internal node:
        # If int, then consider min_samples_split as the minimum number.
        # If float, then min_samples_split is a percentage and ceil(min_samples_split * n_samples)
        # are the minimum number of samples for each split.
        # TODO Create kFold&Scoring PARAM choose
        self.rf_grid = GridSearchCV(self.deopt_classifier, self.paramgrid, cv=self.kfold, refit=True,
                                    n_jobs=-1,
                                    verbose=10,
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


class CatBoostReg(BaseModel):
    model_type = 'CatBoostReg'
    model_category = 'Classification'
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

    def gridsearch(self, X_train, y_train, splits: int = 5, kfold_shuffle: bool = True,
                   scoring_type: str = 'Poisson'):
        print('Gridsearching CatBoost Regressor...')

        if scoring_type in self.eval_metric_map:
            scoring = self.eval_metric_map[scoring_type]
        else:
            print('Scoring method for CatBoost not found in mapping. Defaulting to "Poisson"')
            scoring = 'Poisson'

        self.deopt_classifier = cb.CatBoostRegressor(loss_function='RMSE', random_seed=42, eval_metric=scoring,
                                                     early_stopping_rounds=42)
        #https://towardsdatascience.com/5-cute-features-of-catboost-61532c260f69
        self.paramgrid = {'learning_rate': [0.03, 0.1],
                          'depth': [2,3,4, 6, 8] #DEFAULT is 6. Decrease value to prevent overfitting
                          ,'l2_leaf_reg': [3, 5, 7, 9, 12, 13] #Increase the value to prevent overfitting DEFAULT is 3
                          }

        self.kfold = KFold(n_splits=splits, shuffle=kfold_shuffle, random_state=42)

        self.cb_grid = self.deopt_classifier.grid_search(self.paramgrid, X_train, y_train, cv=self.kfold,
                                                         calc_cv_statistics=True, refit=True, verbose=10, shuffle=False,
                                                         log_cout=sys.stdout,
                                                         log_cerr=sys.stderr
                                                         )
        print("Best Estimator: \n{}\n".format(self.cb_grid['params']))
        self.optimised_model = self.deopt_classifier

        self.best_score = max(self.cb_grid['cv_results']['test-' + str(scoring)+ '-mean'])
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
