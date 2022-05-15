"""'

1. Gather Data
2. Build Model
3. Is my model accurate?
-- No
3a. Measure the uncertainty of predictions
3b. Query for labels -> Return to 2.
-- Yes
4. Employ
'"""
import math
import os
import platform
from datetime import datetime
from random import shuffle
from typing import Callable, Union, Optional

import numpy as np
import pandas as pd
from sqlalchemy.orm.loading import instances
from Data_Analysis import Data_Analyse
import Similarity_Measure
from BatchMode_Committee import batch_sampling
from Cluster import KMeans_Cluster, HDBScan
from CommitteeClass import CommitteeRegressor, CommitteeClassification
from DataLoad import Data_Load_Split
from DiversitySampling_Clustering import DiversitySampling
from Models import SvmModel, RfModel, CBModel, SVR_Model, RandomForestEnsemble, CatBoostReg, SVRLinear
from PreProcess import MinMaxScaling, Standardisation
from QueriesCommittee import max_disagreement_sampling, max_std_sampling
from SelectionFunctions import SelectionFunction
from TrainModel import TrainModel
from confusion_matrix_custom import make_confusion_matrix

"""'

GATHER DATA

'"""

if platform.system() == 'Windows':
    os.chdir()
elif platform.system() == 'Darwin':
    wrk_path_3 = r"/Users/calvin/Documents/OneDrive/Documents/2020/Liposomes Vitamins/LiposomeFormulation"
    os.chdir(wrk_path_3)

classification_output_path_1 = r"/Users/calvin/Documents/OneDrive/Documents/2022/ClassifierCommittee_Output"
classification_output_path_2 = r"C:\Users\Calvin\OneDrive\Documents\2022\ClassifierCommittee_Output"

regression_output_path_1 = r"/Users/calvin/Documents/OneDrive/Documents/2022/RegressorCommittee_Output"
regression_output_path_2 = r"C:\Users\Calvin\OneDrive\Documents\2022\RegressorCommittee_Output"
'''
Pull in Unlabelled data
'''


# TODO need to determine how to implement it into a class (is it even required?)

def unlabelled_data(file, method):
    ul_df = pd.read_csv(file)
    column_drop = ['Duplicate_Check',
                   'PdI Width (d.nm)',
                   'PdI',
                   'Z-Average (d.nm)',
                   'ES_Aggregation']

    useless_clm_drop = ['Req_Weight_1', 'Ethanol_1',
                        'Req_Weight_2', 'Ethanol_2',
                        'Req_Weight_3',
                        'Req_Weight_4', 'Ethanol_4',
                        'ethanol_dil',
                        'component_1_vol_stock',
                        'component_2_vol_stock',
                        'component_3_vol_stock'
                        ]
    ul_df = ul_df.drop(columns=column_drop)
    ul_df = ul_df.drop(columns=useless_clm_drop)
    ul_df.replace(np.nan, 'None', inplace=True)
    ul_df = pd.get_dummies(ul_df, columns=["Component_1", "Component_2", "Component_3"],
                           prefix="", prefix_sep="")
    # if method=='fillna':    ul_df['Component_3'] = ul_df['Component_3'].apply(lambda x: None if pd.isnull(x) else x) #TODO This should be transformed into an IF function, thus when the function for unlabelled is filled with a parameter, then activates

    ul_df = ul_df.groupby(level=0, axis=1, sort=False).sum()

    # print(ul_df.isna().any())
    X_val = ul_df.to_numpy()
    columns_x_val = ul_df.columns
    return X_val, columns_x_val


'''Split the Data'''

'''

Selection

'''


class Algorithm(object):
    accuracies = []

    def __init__(self, model_object, model_type,
                 save_path: Optional[str],
                 scoring_type: str,
                 hide: str = None,
                 select: Callable = SelectionFunction.entropy_sampling,
                 file: str = 'unlabelled_data_full.csv',
                 limit: int = -1, perc_uncertain: float = 0.1, n_instances: Union[int, str] = 'Full',
                 split_ratio: float = 0.2
                 ):
        self.regression = None
        self.similarity_score = None
        self.selection_probas_val = None
        assert perc_uncertain <= 1
        assert model_type in ['Classification', 'Regression']
        self.today_date = datetime.today().strftime('%Y%m%d')
        self.today_date_time = datetime.today().strftime('%Y%m%d_%H%M_incomplete')
        self.model_test = None
        self.optimised_classifier = None
        self.model_object = model_object
        self.selection_functions = SelectionFunction()
        self.diversity_sampling = DiversitySampling()
        self.select = select
        self.file = file
        self.limit = limit
        self.n_instances = n_instances
        self.split_ratio = split_ratio
        self.target_labels = None
        self.perc_uncertain = perc_uncertain
        self.models_algorithms = None
        self.model_type = model_type
        self.hide = hide
        self.classification = None
        self.scoring_type = scoring_type
        self.save_path = save_path

        #######New Data - from self.create_data()/self.normalise_data()
        self.X_val = None
        self.y_train = None
        self.X_train = None
        self.y_test = None
        self.X_test = None
        self.normaliser = None

        ####Clustering
        self.best_k = None
        self.results = None
        ##############Things to run
        self.create_save_folder()
        self.create_data()
        self.normalise_data()

    def create_save_folder(self):
        dirName = self.today_date_time

        try:
            # Create Directory
            save_folder = os.path.join(self.save_path, dirName)
            os.mkdir(save_folder)
            print("Directory ", dirName, " Created ")
            self.save_path = save_folder
        except FileExistsError:
            print("Directory ", dirName, " already exists")

    def create_data(self):

        # Pull in unlabelled Data

        self.X_val, self.columns_x_val = unlabelled_data(self.file,
                                                         method='fillna')  # TODO need to rename

        if self.limit > 0:
            shuffle(self.X_val)
            self.X_val = self.X_val[:self.limit]
        else:
            shuffle(self.X_val)

        if isinstance(self.n_instances, int):
            self.n_instances = self.n_instances
        elif isinstance(self.n_instances, str):
            if self.n_instances in ['Full', 'full', 'All', 'all']:
                self.n_instances = len(self.X_val)
            elif self.n_instances in ['Half', 'half']:
                self.n_instances = round(len(self.X_val) / 2)

        # uncertain_count = math.ceil(len(self.X_val) * self.perc_uncertain)

        # Run Split

        data_load_split = Data_Load_Split("Results_Complete.csv", hide_component=self.hide, alg_categ=self.model_type,
                                          split_ratio=self.split_ratio,
                                          shuffle_data=True)
        self.converted_columns = data_load_split.columns_converted
        self.target_labels = data_load_split.class_names_str

        self.X_train, self.X_test, self.y_train, self.y_test = data_load_split.split_train_test()
        return self.X_train, self.X_test, self.y_train, self.y_test
        # normalise data - Increased accuracy from 50% to 89%

    def normalise_data(self):
        self.normaliser = MinMaxScaling()
        self.normaliser.fit_scale(self.X_train, self.X_test, self.X_val, self.converted_columns)
        self.X_train, self.X_val, self.X_test = self.normaliser.transform_scale(self.X_train, self.X_val, self.X_test, converted_columns=self.converted_columns)
        return self.X_train, self.X_val, self.X_test

    def analyse_data(self):
        data_analyse = Data_Analyse()

        data_analyse.histogram(self.y_train,data_name='y_train', save_path=self.save_path,plot=False)
        data_analyse.histogram(self.y_test,data_name='y_test', save_path=self.save_path,plot=False)

        data_analyse.qqplot_data(self.y_train,data_name='y_train',save_path=self.save_path ,plot=False)
        data_analyse.qqplot_data(self.y_test,data_name='y_test',save_path=self.save_path ,plot=False)

        print('Shapiro Wilk Y Train')
        self.shapiro_wilk_y_train = data_analyse.shapiro_wilk_test(self.y_train)

        print(self.shapiro_wilk_y_train)
        print('Shapiro Wilk Y Test')
        self.shapiro_wilk_y_test = data_analyse.shapiro_wilk_test(self.y_test)

        print(self.shapiro_wilk_y_test)

        print('Dagostino K^2 Y Train')
        self.dagostino_k2_y_train, self.dagostino_p_y_train = data_analyse.dagostino_k2(self.y_train)
        print('Dagostino K^2 Y Test')
        self.dagostino_k2_y_test , self.dagostino_p_y_test = data_analyse.dagostino_k2(self.y_test)


        print('Anderson Y Train')
        self.anderson_darling_train = data_analyse.anderson_darling(self.y_train)
        print('Anderson Y Test')
        self.anderson_darling_test = data_analyse.anderson_darling(self.y_test)

        print('Heatmap X Train')
        self.heatmap_train = data_analyse.heatmap(self.X_train, self.columns_x_val,data_name='x_train',save_path=self.save_path ,plot=False)

        print('Box Plot X Train')
        self.box_plot_train = data_analyse.box_plot(self.X_train, self.columns_x_val,data_name='x_train',save_path=self.save_path ,plot=False)

        print('Variance Inflation Factor_X_train')
        self.variance_inflation_factor_x_train = data_analyse.variance_inflation_factor(self.X_train,self.columns_x_val)
        print(self.variance_inflation_factor_x_train)

        return




    def run_algorithm(self, splits: int = 5, grid_params=None, skip_unlabelled_analysis: bool = False, verbose: int = 0):
        if self.model_type == 'Regression':
            self.regression_model(splits, grid_params, skip_unlabelled_analysis=skip_unlabelled_analysis, verbose=verbose)
            self.regression = 1
        elif self.model_type == 'Classification':
            self.classification_model(splits, grid_params)
            self.classification = 1

    def regression_model(self, splits: int = 5, grid_params=None, skip_unlabelled_analysis: bool = False,
                         verbose: int = 0):
        if type(self.model_object) is list:
            self.committee_models = CommitteeRegressor(self.model_object, self.X_train, self.X_test, self.y_train,
                                                       self.y_test, self.X_val,
                                                       splits=splits,
                                                       kfold_shuffle=True,
                                                       scoring_type=self.scoring_type,
                                                       instances=self.n_instances,
                                                       query_strategy=max_std_sampling)
            self.scores = self.committee_models.gridsearch_committee(grid_params=grid_params, verbose=verbose)
            self.committee_models.fit_data()
            self.score_data = self.committee_models.score()
            self.selection_probas_val, *rest = self.committee_models.query(self.committee_models,
                                                                           n_instances=self.n_instances)
            # test = batch_sampling(models=self.committee_models, X=self.X_val, X_labelled=self.X_train,
            #                      converted_columns=self.converted_columns, query_type=max_std_sampling, n_jobs=-1,
            #                      metric='gower')
            self.committee_models.predictionvsactual(save_path=self.save_path, plot=False)
            self.models_algorithms = self.committee_models.printname()
            self.committee_models.lime_analysis(self.columns_x_val, save_path=self.save_path,
                                                skip_unlabelled_analysis=skip_unlabelled_analysis)

    def classification_model(self, splits: int = 5, grid_params=None):
        if type(self.model_object) is list:
            self.committee_models = CommitteeClassification(learner_list=self.model_object, X_training=self.X_train,
                                                            X_testing=self.X_test,
                                                            y_training=self.y_train, y_testing=self.y_test,
                                                            X_unlabeled=self.X_val,
                                                            query_strategy=max_disagreement_sampling,
                                                            c_weight='balanced',
                                                            splits=splits,
                                                            scoring_type='precision',
                                                            kfold_shuffle=True)
            self.scores = self.committee_models.gridsearch_committee(grid_params=grid_params)
            self.committee_models.fit_data()
            # self.probas_val = self.committee_models.vote_proba()

            self.selection_probas_val = self.committee_models.query(self.committee_models, n_instances=self.n_instances,
                                                                    X_labelled=self.X_train)

            self.conf_matrix = self.committee_models.confusion_matrix()
            self.precision_scores = self.committee_models.precision_scoring()
            self.models_algorithms = self.committee_models.printname()
            self.committee_models.lime_analysis(self.columns_x_val, save_path=self.save_path)

    def compare_query_changes(self):
        selection_df = pd.DataFrame(self.selection_probas_val[1]).reset_index(drop=True)
        unlabelled_df_temp = pd.DataFrame(self.X_val.copy()).reset_index(drop=True)
        print(selection_df.equals(unlabelled_df_temp))

    def similairty_scoring(self, method: str = 'gower', threshold: float = 0.5, n_instances: int = 100,
                           k_range: Optional[int] = 10,
                           alpha: Optional[float] = 0.01):
        self.density_method = method
        self.method_threshold = threshold
        sim_init = Similarity_Measure.Similarity(self.selection_probas_val[1], self.selection_probas_val[0])

        if method == 'cosine':
            assert threshold <= 1.0
            self.samples, self.samples_index, self.sample_score = sim_init.similarity_cosine(threshold, 'cosine')
        elif method == 'gower':
            assert threshold <= 1.0
            self.samples, self.samples_index, self.sample_score = sim_init.similarity_gower(threshold,
                                                                                            n_instances=n_instances,
                                                                                            converted_columns=self.converted_columns)
        elif method == 'kmeans':
            self.kmeans_cluster_deopt = KMeans_Cluster(unlabeled_data=self.selection_probas_val)

            self.best_k, self.results = self.kmeans_cluster_deopt.chooseBestKforKmeansParallel(k_range=k_range,
                                                                                               alpha=alpha)

            self.kmeans_cluster_opt = KMeans_Cluster(unlabeled_data=self.selection_probas_val, n_clusters=self.best_k)
            self.kmeans_cluster_opt.kmeans_fit()
            self.samples, self.samples_index, self.sample_score = self.kmeans_cluster_opt.create_array(
                threshold=threshold,
                n_instances=n_instances)


        elif method == 'hdbscan':
            self.hdbscan_opt = HDBScan(unlabeled_data=self.selection_probas_val)
            self.hdbscan_opt.hdbscan_fit()

            self.samples, self.samples_index, self.sample_score = self.hdbscan_opt.distance_sort()

        return self.samples, self.samples_index, self.sample_score

    def single_model(self):
        # TODO Needs to be cleaned up and rectified
        ### GridSearch/Fit Data - Single Algorithm ###
        self.model_test = TrainModel(self.model_object)
        self.optimised_classifier = self.model_test.optimise(self.X_train, self.y_train, 'balanced', splits=5,
                                                             scoring='precision')
        (X_train, X_val, X_test) = self.model_test.train(self.X_train, self.y_train, self.X_val, self.X_test)
        probas_val = \
            self.model_test.predictions(X_train, X_val, X_test)
        self.model_test.return_accuracy(1, self.y_test, self.y_train)
        model_type = self.model_object.model_type
        ##Attempt to get PROBA values of X_Val
        randomize_tie_break = True
        selection_probas_val = \
            self.selection_functions.select(self.optimised_classifier, X_val, instances, randomize_tie_break)

        if self.select == 'margin_sampling':
            selection_probas_val = \
                self.selection_functions.margin_sampling(self.optimised_classifier, X_val, instances,
                                                         randomize_tie_break)
        elif self.select == 'entropy_sampling':
            selection_probas_val = \
                self.selection_functions.entropy_sampling(self.optimised_classifier, X_val, instances,
                                                          randomize_tie_break)
        elif self.select == 'uncertainty_sampling':
            selection_probas_val = \
                self.selection_functions.uncertainty_sampling(self.optimised_classifier, X_val, instances,
                                                              randomize_tie_break)

    def output_data(self):
        # print(self.probas_val)

        # print('probabilities:', self.probas_val.shape, '\n',
        #      np.argmax(self.probas_val, axis=1))
        # print('the unique values in the probability values is: ', np.unique(self.probas_val))
        # print('size of unique values in the probas value array: ', np.unique(self.probas_val).size)

        # print('SHAPE OF X_TRAIN', self.X_train.shape[0])
        # print(self.optimised_classifier.classes_)
        # print(selection_probas_val)
        # similarity_scores = []
        # index = []
        # data_info = []
        # for _, data in enumerate(samples):
        #    for _, data_x in enumerate(data):
        #        similarity_scores.append(data_x[0])
        #        index.append(data_x[1])
        #        data_info.append(data_x[2])

        _, reversed_x_val, _ = self.normaliser.inverse(self.X_train, self.samples, self.X_test,
                                                       converted_columns=self.converted_columns)

        df = pd.DataFrame(reversed_x_val, columns=self.columns_x_val, index=self.samples_index)
        # add similarty scores:
        self.sample_score.index = df.index
        df['sample_scoring'] = self.sample_score
        data = {'date': self.today_date,
                'model_type': self.model_type}
        if type(self.model_object) is list:
            if len(self.model_object) == 2:
                data['algorithm_1'] = str(self.model_object[0].model_type)
                data['algorithm_2'] = str(self.model_object[1].model_type)
            elif len(self.model_object) == 3:
                data['algorithm_1'] = str(self.model_object[0].model_type)
                data['algorithm_2'] = str(self.model_object[1].model_type)
                data['algorithm_3'] = str(self.model_object[2].model_type)
        else:
            data['algorithm_1'] = str(self.model_object[0].model_type)

        data['density_method'] = self.density_method
        data['method_threshold'] = self.method_threshold

        df_info = pd.DataFrame.from_dict(data, orient='index').T
        col_move = len(df_info.columns)
        df_scores = pd.DataFrame.from_dict(self.scores, orient='index').T
        df_scores['average_score'] = df_scores[1:].sum(axis=1) / len(self.model_object)

        file_name = str(self.models_algorithms) + '_Ouput_Selection_' + str(self.select.__name__) + '_' + str(
            self.today_date) + '.xlsx'
        file_name = os.path.join(self.save_path, file_name)

        writer = pd.ExcelWriter(file_name
                                ,
                                engine='xlsxwriter')
        df_info.to_excel(writer,
                         index=False, startrow=1)
        df_scores.to_excel(writer,
                           index=False, startcol=col_move)

        df.to_excel(
            writer,
            index=True, startrow=13)

        if self.classification == 1:
            if os.name == 'posix':
                os.chdir(classification_output_path_1)
                print("Utilising MacBook")
            else:
                os.chdir(classification_output_path_2)
                print("Utilising Home Pathway")
            labels = ["True Neg", "False Pos", "False Neg", "True Pos"]
            categ = ["Zero", "One"]
            df_precision_score = pd.DataFrame.from_dict(self.precision_scores, orient='Index')
            df_precision_score.to_excel(writer,
                                        index=True,
                                        startrow=3)
            for k, v in self.conf_matrix.items():
                print("Building out confusion matrix for: " + str(k))
                plot = make_confusion_matrix(v, group_names=labels,
                                             categories=categ,
                                             cmap="Blues",
                                             title=str(k))
                fig = plot.get_figure()
                fig.savefig(str(k) + ".png", dpi=400)
                temp_df = pd.DataFrame()
                temp_df.to_excel(writer, sheet_name=str(k))
                worksheet = writer.sheets[str(k)]
                worksheet.insert_image('C2', str(k) + ".png")

        elif self.regression == 1:
            df_scores_data = pd.DataFrame.from_dict(self.score_data, orient='index').T
            df_scores_data.to_excel(writer, sheet_name='Regression_Score_Data')
            if os.name == 'posix':
                os.chdir(regression_output_path_1)
                print("Utilising MacBook")
            else:
                os.chdir(regression_output_path_2)
                print("Utilising Home Pathway")

        writer.save()

    def change_folder_name_complete(self):

        old_name = self.save_path
        new_name = old_name.replace('incomplete', 'complete')
        os.rename(old_name, new_name)


# models = [SvmModel, RfModel, CBModel]
models = [SVRLinear, RandomForestEnsemble, CatBoostReg]
# selection_functions = [selection_functions]

SvmModel = {'C': [0.01, 0.1],  # np.logspace(-5, 2, 8),
            'gamma': np.logspace(-5, 3, 9),
            # Hashing out RBF provides more variation in the proba_values, however, the uniqueness 12 counts
            'kernel': ['rbf', 'poly', 'sigmoid', 'linear'],
            'coef0': [0, 0.001, 0.1, 1],
            'degree': [1, 2, 3, 4]}

RfModel = {'n_estimators': [int(x) for x in np.linspace(start=200, stop=2000, num=10)],
           #                  'max_features': ['auto', 'sqrt'],
           #                  'max_depth': [int(x) for x in np.linspace(10, 35, num=11)]
           # ,'min_samples_split': [2, 5, 10]
           # ,'min_samples_leaf': [1, 2, 4]
           # ,'bootstrap': [True, False]
           }

CBModel = {
    'depth': [3, 1, 2, 6, 4, 5, 7, 8, 9, 10]
    #    , 'n_estimators': [250, 100, 500, 1000]
    #    , 'learning_rate': [0.03, 0.001, 0.01, 0.1, 0.2, 0.3]
    #    , 'l2_leaf_reg': [3, 1, 5, 10, 100],
    #    'border_count': [32, 5, 10, 20, 50, 100, 200]
}
# https://www.vebuso.com/2020/03/svm-hyperparameter-tuning-using-gridsearchcv/
SVM_Reg = {'C': [50, 100, 150, 200, 500, 1000, 5000, 10000],  # np.logspace(-6, 3,10), #*
           'gamma': np.logspace(-4, 2, 7),  # Kernel coefficient for rbf, poly & sig
           # Hashing out RBF provides more variation in the proba_values, however, the uniqueness 12 counts
           'kernel': [  # 'rbf',
               # 'poly',
               # 'sigmoid'
               'linear'  # *
           ]
           # ,'coef0': [0.1, 1] #Independent term for poly & sig
           # 'degree': [1, 3, 4], #Poly
    , 'epsilon': [0.1, 0.2, 0.3, 0.5, 1, 5, 10, 50, 70]
    , 'tol': np.logspace(-6, 1, 8)
           }
RFE_Reg = {'n_estimators': [int(x) for x in np.linspace(start=200,
                                                        stop=1000,
                                                        num=9)],
           # 'criterion': ['squared_error', 'absolute_error', 'poisson'],
           # 'max_features': ['auto', 'sqrt', 'log2'],
           # Hashing out RBF provides more variation in the proba_values, however, the uniqueness 12 counts
           # 'max_depth': [int(x) for x in np.linspace(1, 110,num=12)],
           # 'bootstrap': [False, True],
           # 'min_samples_leaf': [float(x) for x in np.arange(0.1, 0.6, 0.1)]
           }
CatBoost_Reg = {'learning_rate': [0.03, 0.1],
                'depth': [2, 3, 4, 6]  # DEFAULT is 6. Decrease value to prevent overfitting
    , 'l2_leaf_reg': [3, 12, 13]  # Increase the value to prevent overfitting DEFAULT is 3
                }

SVRLinear = {'C': [50, 100, 150, 200, 500, 1000, 5000, 10000],
             'tol':  np.logspace(-6, 1, 8),
             'epsilon': [0.1, 0.2, 0.3, 0.5, 1, 5, 10, 50, 70],
             'loss': ['epsilon_insensitive', 'squared_epsilon_insensitive'],
             'fit_intercept': [True, False],
             'max_iter': [1000, 10000]


}
grid_params = {}
grid_params['SVC'] = SvmModel
grid_params['Random_Forest'] = RfModel
grid_params['CatBoostClass'] = CBModel
grid_params['SVR'] = SVM_Reg
grid_params['RFE_Regressor'] = RFE_Reg
grid_params['CatBoostReg'] = CatBoost_Reg
grid_params['SVR_Linear'] = SVRLinear

save_path = regression_output_path_1
alg = Algorithm(models, select=max_std_sampling, model_type='Regression',
                scoring_type='r2', save_path=save_path, split_ratio=0.3)

alg.analyse_data()
alg.run_algorithm(splits=5, grid_params=grid_params, skip_unlabelled_analysis=True, verbose=10)
alg.compare_query_changes()
alg.similairty_scoring(method='gower', threshold=0.25, n_instances=100)
alg.output_data()
alg.change_folder_name_complete()
