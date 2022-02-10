import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
import catboost as cb
from BaseModel import BaseModel


class SvmModel(BaseModel):
    model_type = 'SVC'

    def __init__(self):
        self.paramgrid = None
        self.proba = None
        self.optimised_model = None
        self.val_y_predicted = None
        self.test_y_predicted = None
        self.stratifiedkfold = None
        self.deopt_classifier = None
        self.train_y_predicted = None

    def gridsearch(self, X_train, y_train, c_weight, splits: int = 5, scoring_type: str = 'precision'):
        print("GridSearching SVM...")
        self.deopt_classifier = SVC(class_weight=c_weight, random_state=42, probability=True)
        self.stratifiedkfold = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)
        # TODO Better define params
        self.paramgrid = {'C': [0.01],  # np.logspace(-5, 2, 8),
                          'gamma': np.logspace(-5, 3, 9),
                          # Hashing out RBF provides more variation in the proba_values, however, the uniqueness 12 counts
                          'kernel': [  # 'rbf',
                              'poly', 'sigmoid', 'linear'],
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
        return self.optimised_model

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
        self.test_y_predicted = self.optimised_model(X_test)

        return self.train_y_predicted, self.test_y_predicted

    def predict_proba(self, X_val):
        print('Proba prediction...')
        self.proba = self.optimised_model.predict_proba(X_val)
        return self.proba


class RfModel(BaseModel):
    model_type = 'Random_Forest'

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

    def gridsearch(self, X_train, y_train, c_weight, splits: int = 5, scoring: str = 'precision'):
        print('Gridsearching RFModel...')
        self.deopt_classifier = RandomForestClassifier(class_weight=c_weight)
        self.stratifiedkfold = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)
        self.paramgrid = {'n_estimators': [int(x) for x in np.linspace(start=200, stop=2000, num=10)],
                          # 'max_features': ['auto', 'sqrt'],
                          # 'max_depth': [int(x) for x in np.linspace(10, 110, num=11)],
                          # 'min_samples_split': [2, 5, 10],
                          # 'min_samples_leaf': [1, 2, 4],
                          'bootstrap': [True, False]}

        self.rf_grid = GridSearchCV(self.deopt_classifier, self.paramgrid, cv=self.stratifiedkfold, refit=True,
                                    scoring=scoring,
                                    verbose=10, n_jobs=-1)

        self.rf_grid.fit(X_train, y_train)
        print("Best Estimator: \n{}\n".format(self.rf_grid.best_estimator_))
        print("Best Parameters: \n{}\n".format(self.rf_grid.best_params_))
        print("Best Test Score: \n{}\n".format(self.rf_grid.best_score_))

        self.optimised_model = self.rf_grid.best_estimator_
        return self.optimised_model

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
        self.test_y_predicted = self.optimised_model(X_test)

        return self.train_y_predicted, self.test_y_predicted

    def predict_proba(self, X_val):
        print('Proba prediction...')
        self.proba = self.optimised_model.predict_proba(X_val)
        return self.proba


class CBModel(BaseModel):
    model_type = 'CatBoost'

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

    def gridsearch(self, X_train, y_train, c_weight, splits: int = 5, scoring: str = 'Precision'):
        print('GridSearching CatBoost...')
        self.deopt_classifier = cb.CatBoostClassifier(loss_function='Logloss', eval_metric=scoring,
                                                      auto_class_weights=c_weight,
                                                      random_seed=42)
        self.stratifiedkfold = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)
        self.paramgrid = {'depth': [3, 1, 2, 6, 4, 5, 7, 8, 9, 10],
                          'iterations': [250, 100, 500, 1000],
                          'learning_rate': [0.03, 0.001, 0.01, 0.1, 0.2, 0.3],
                          'l2_leaf_reg': [3, 1, 5, 10, 100],
                          'border_count': [32, 5, 10, 20, 50, 100, 200]}
        self.cb_grid = GridSearchCV(self.deopt_classifier, self.paramgrid, cv=self.stratifiedkfold, refit=True,
                                    verbose=10, n_jobs=-1)
        self.cb_grid.fit(X_train, y_train)
        print("Best Estimator: \n{}\n".format(self.cb_grid.best_estimator_))
        print("Best Parameters: \n{}\n".format(self.cb_grid.best_params_))
        print("Best Test Score: \n{}\n".format(self.cb_grid.best_score_))

        self.optimised_model = self.cb_grid.best_estimator_
        return self.optimised_model

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
        self.test_y_predicted = self.optimised_model(X_test)

        return self.train_y_predicted, self.test_y_predicted

    def predict_proba(self, X_val):
        print('Proba prediction...')
        self.proba = self.optimised_model.predict_proba(X_val)
        return self.proba
