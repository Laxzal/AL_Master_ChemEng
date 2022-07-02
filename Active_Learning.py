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
import random
from datetime import datetime
from random import shuffle
from typing import Callable, Union, Optional
import shortuuid
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
from Models import SvmModel, RfModel, CBModel, SVR_Model, RandomForestEnsemble, CatBoostReg, SVRLinear, Neural_Network
from PreProcess import MinMaxScaling, Standardisation
from QueriesCommittee import max_disagreement_sampling, max_std_sampling
from SelectionFunctions import SelectionFunction
from TrainModel import TrainModel
from confusion_matrix_custom import make_confusion_matrix

from mrmr_algorithm import MRMR

"""'

GATHER DATA

'"""

if platform.system() == 'Windows':
    os.chdir()
elif platform.system() == 'Darwin':
    wrk_path_3 = r"/Users/calvin/Documents/OneDrive/Documents/2022/Data_Output"
    os.chdir(wrk_path_3)

classification_output_path_1 = r"/Users/calvin/Documents/OneDrive/Documents/2022/ClassifierCommittee_Output"
classification_output_path_2 = r"C:\Users\Calvin\OneDrive\Documents\2022\ClassifierCommittee_Output"

regression_output_path_1 = r"/Users/calvin/Documents/OneDrive/Documents/2022/RegressorCommittee_Output"
regression_output_path_2 = r"C:\Users\Calvin\OneDrive\Documents\2022\RegressorCommittee_Output"
'''
Pull in Unlabelled data
'''


# TODO need to determine how to implement it into a class (is it even required?)

def unlabelled_data(file, method, column_removal_experiment: list=None):
    ul_df = pd.read_csv(file)
    column_drop = ['Duplicate_Check',
                   'PdI Width (d.nm)',
                   'PdI',
                   'Z-Average (d.nm)',
                   'ES_Aggregation']


    ul_df = ul_df.drop(columns=column_drop)
    #ul_df = ul_df.drop(columns=useless_clm_drop)

    #Remove a column(s)
    if column_removal_experiment is not None:
        ul_df.drop(columns=column_removal_experiment, inplace=True)


    ul_df.replace(np.nan, 'None', inplace=True)
    if "Component_1" and "Component_2" and "Component_3" in ul_df.columns:
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
                 split_ratio: float = 0.2,
                 column_removal_experiment: list=None,
                 MRMR_K_Value: int=10
                 ):
        self.regression = None
        self.similarity_score = None
        self.selection_probas_val = None
        assert perc_uncertain <= 1
        assert model_type in ['Classification', 'Regression']
        self.today_date = datetime.today().strftime('%Y%m%d')
        self.today_date_time = datetime.today().strftime('%Y%m%d_%H%M_incomplete')
        self.time = datetime.today().strftime('%H%M')
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
        self.column_removal_experiment=column_removal_experiment
        self.uuid = shortuuid.ShortUUID().random(length=10).upper()
        self.K = MRMR_K_Value
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
        dirName = str(self.uuid) +'_'+ str(self.today_date_time)

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
                                                         method='fillna', column_removal_experiment=self.column_removal_experiment)  # TODO need to rename
        random.seed(42) #This shhould ensure shuffling is always the same
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
        #DLS onlly accurate up to 1000.0 size, need to filter

        data_load_split = Data_Load_Split("Results_Complete.csv", hide_component=self.hide, alg_categ=self.model_type,
                                          split_ratio=self.split_ratio,
                                          shuffle_data=True, filter_target=True, target='Z-Average (d.nm)',
                                          smaller_than=1000.0, column_removal_experiment=self.column_removal_experiment)
        self.converted_columns = data_load_split.columns_converted
        self.target_labels = data_load_split.class_names_str

        self.gaussian_result = data_load_split.analyse_data(save_path=self.save_path, column_names=self.columns_x_val, plot = False)
        mrmr = MRMR(data_load_split.X, data_load_split.y, self.columns_x_val, K=self.K)
        self.selected, self.not_selected = mrmr.computing_correlations()

        self.X_train, self.X_test, self.y_train, self.y_test = data_load_split.split_train_test()


        ### Log Transform y target
        #TODO Implement better

        if self.gaussian_result == False:
            self.y_train = np.log(self.y_train)
            self.y_test = np.log(self.y_test)
        else:
            pass
        ###
        return self.X_train, self.X_test, self.y_train, self.y_test
        # normalise data - Increased accuracy from 50% to 89%

    def normalise_data(self):
        self.normaliser = MinMaxScaling()
        self.normaliser.fit_scale(self.X_train, self.X_test, self.X_val, self.converted_columns)
        self.X_train, self.X_val, self.X_test = self.normaliser.transform_scale(self.X_train, self.X_val, self.X_test, converted_columns=self.converted_columns)
        return self.X_train, self.X_val, self.X_test



    def run_algorithm(self, initialisation: str= 'gridsearch', splits: int = 5, grid_params=None, skip_unlabelled_analysis: bool = False, verbose: int = 0, kfold_repeats: int =1):
        self.kfold_splits = splits
        if self.model_type == 'Regression':
            self.regression_model(initialisation, splits, grid_params, skip_unlabelled_analysis=skip_unlabelled_analysis, verbose=verbose, kfold_shuffle=kfold_repeats)
            self.regression = 1
        elif self.model_type == 'Classification':
            self.classification_model(splits, grid_params)
            self.classification = 1


    def regression_model(self, initialisation: str = 'gridsearch', splits: int = 5, grid_params=None, skip_unlabelled_analysis: bool = False,
                         verbose: int = 0, kfold_shuffle: int = 1):
        if type(self.model_object) is list:
            self.committee_models = CommitteeRegressor(self.model_object, self.X_train, self.X_test, self.y_train,
                                                       self.y_test, self.X_val,
                                                       splits=splits,
                                                       kfold_shuffle=kfold_shuffle,
                                                       scoring_type=self.scoring_type,
                                                       instances=self.n_instances,
                                                       query_strategy=max_std_sampling)
            if initialisation == 'gridsearch':
                self.scores = self.committee_models.gridsearch_committee(initialisation=initialisation, grid_params=grid_params, verbose=verbose)
            elif initialisation == 'optimised':
                self.scores = self.committee_models
            self.committee_models.fit_data()
            self.score_data = self.committee_models.score()
            self.selection_probas_val, *rest = self.committee_models.query(self.committee_models,
                                                                           n_instances=self.n_instances)
            # test = batch_sampling(models=self.committee_models, X=self.X_val, X_labelled=self.X_train,
            #                      converted_columns=self.converted_columns, query_type=max_std_sampling, n_jobs=-1,
            #                      metric='gower')
            self.committee_models.predictionvsactual(save_path=self.save_path, plot=False)
            self.models_algorithms = self.committee_models.printname()
            self.committee_models.out_cv_score(save_path=self.save_path)
            #self.committee_models.lime_analysis(self.columns_x_val, save_path=self.save_path,
            #                                   skip_unlabelled_analysis=skip_unlabelled_analysis)

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
        self.method_n_instances = n_instances
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

    def random_unlabelled(self, n_instances: int):




        _, reversed_x_val, _ = self.normaliser.inverse(self.X_train, self.X_val, self.X_test,
                                                       converted_columns=self.converted_columns)
        unlabeled_data= pd.DataFrame(reversed_x_val, columns=self.columns_x_val)
        unlabeled_data['original_index'] = unlabeled_data.index

        random_data = unlabeled_data.sample(n=n_instances)
        file_name = "random_unlabeled_data_points.xlsx"
        file_name = os.path.join(self.save_path, file_name)
        random_data.to_excel(file_name, index_label=False)


    def master_file(self):

        path = r"/Users/calvin/Documents/OneDrive/Documents/2022/Data_Output"
        file_name = "MasterFile_AL_Results.xlsx"
        #Check if file exists
        if os.path.isfile(os.path.join(path,file_name)) == True:
            #build out file
            df_file = pd.read_excel(os.path.join(path,file_name))
            df_temp = pd.DataFrame(columns=df_file.columns)

            df_temp['date'] = [int(self.today_date)]
            df_temp['time'] = [int(self.time)]
            df_temp['guid'] = [self.uuid]

            temp_list = []
            for i in range(0,len(self.model_object)):
                temp_list.append(str(self.model_object[i].model_type))

            listToStr = '-'.join(map(str, temp_list))
            df_temp['algorithms'] = [listToStr]
            df_temp['sampling_method'] = self.select.__name__
            df_temp['model_type'] = self.model_type
            df_temp['scoring_type'] = self.scoring_type
            df_temp['split_ratio'] = [float(self.split_ratio)]
            temp_list_columns=[]
            for i in range(0,len(self.column_removal_experiment)):
                temp_list_columns.append(str(self.column_removal_experiment[i]))

            listToStr_col = ', '.join(map(str, temp_list_columns))

            df_temp['columns_removed'] = [listToStr_col]
            df_temp['kfold_splits'] = self.kfold_splits
            df_temp['normaliser'] = str(self.normaliser.name)
            df_temp['alg_1'] = [self.scores[str(self.model_object[0].model_type)][0]]
            df_temp['alg_1_cv'] = self.scores[str(self.model_object[0].model_type)][1]
            df_temp['alg_1_train'] = self.score_data[str(self.model_object[0].model_type) + '_train'][1]
            df_temp['alg_1_test'] = self.score_data[str(self.model_object[0].model_type) + '_test'][1]
            df_temp['alg_2'] = [self.scores[str(self.model_object[1].model_type)][0]]
            df_temp['alg_2_cv'] = self.scores[str(self.model_object[1].model_type)][1]
            df_temp['alg_2_train'] = self.score_data[str(self.model_object[1].model_type) + '_train'][1]
            df_temp['alg_2_test'] = self.score_data[str(self.model_object[1].model_type) + '_test'][1]
            df_temp['alg_3'] = [self.scores[str(self.model_object[2].model_type)][0]]
            df_temp['alg_3_cv'] = self.scores[str(self.model_object[2].model_type)][1]
            df_temp['alg_3_train'] = self.score_data[str(self.model_object[2].model_type) + '_train'][1]
            df_temp['alg_3_test'] = self.score_data[str(self.model_object[2].model_type) + '_test'][1]

            df_temp['density_metric'] = [self.density_method]
            df_temp['metric_threshold'] = [self.method_threshold]
            df_temp['n_instances'] = [self.method_n_instances]
            df_temp['MRMR_K'] = [self.K]
            df_temp['MRMR_Selected'] = [self.selected]
            df_temp['MRMR_Not_Selected'] = [self.not_selected]
            master_df = pd.concat([df_file, df_temp],ignore_index=True, axis=0)

        else:
            #create new file
            master_df = pd.DataFrame(columns=['date','time','guid','algorithms','sampling_method','model_type',
                                              'scoring_type','split_ratio','columns_removed','kfold_splits','normaliser',
                                              'alg_1','alg_1_cv','alg_1_train','alg_1_test',
                                              'alg_2','alg_2_cv','alg_2_train','alg_2_test',
                                              'alg_3','alg_3_cv','alg_3_train','alg_3_test','density_metric',
                                              'metric_threshold','n_instances'])

            master_df['date'] = [self.today_date]
            master_df['time'] = [self.time]
            master_df['guid'] = [self.uuid]

            temp_list = []
            for i in range(0,len(self.model_object)):
                temp_list.append(str(self.model_object[i].model_type))

            listToStr = '-'.join(map(str, temp_list))
            master_df['algorithms'] = [listToStr]
            master_df['sampling_method'] = self.select.__name__
            master_df['model_type'] = self.model_type
            master_df['scoring_type'] = self.scoring_type
            master_df['split_ratio'] = self.split_ratio
            temp_list_columns=[]
            for i in range(0,len(self.column_removal_experiment)):
                temp_list_columns.append(str(self.column_removal_experiment[i]))

            listToStr_col = ', '.join(map(str, temp_list_columns))

            master_df['columns_removed'] = [listToStr_col]
            master_df['kfold_splits'] = self.kfold_splits
            master_df['normaliser'] = str(self.normaliser.name)
            master_df['alg_1'] = [self.scores[str(self.model_object[0].model_type)][0]]
            master_df['alg_1_cv'] = self.scores[str(self.model_object[0].model_type)][1]
            master_df['alg_1_train'] = self.score_data[str(self.model_object[0].model_type) + '_train'][1]
            master_df['alg_1_test'] = self.score_data[str(self.model_object[0].model_type) + '_test'][1]
            master_df['alg_2'] = [self.scores[str(self.model_object[1].model_type)][0]]
            master_df['alg_2_cv'] = self.scores[str(self.model_object[1].model_type)][1]
            master_df['alg_2_train'] = self.score_data[str(self.model_object[1].model_type) + '_train'][1]
            master_df['alg_2_test'] = self.score_data[str(self.model_object[1].model_type) + '_test'][1]
            master_df['alg_3'] = [self.scores[str(self.model_object[2].model_type)][0]]
            master_df['alg_3_cv'] = self.scores[str(self.model_object[2].model_type)][1]
            master_df['alg_3_train'] = self.score_data[str(self.model_object[2].model_type) + '_train'][1]
            master_df['alg_3_test'] = self.score_data[str(self.model_object[2].model_type) + '_test'][1]

            master_df['density_metric'] = [self.density_method]
            master_df['metric_threshold'] = [self.method_threshold]
            master_df['n_instances'] = [self.method_n_instances]
            master_df['MRMR_K'] = [self.K]
            master_df['MRMR_Selected'] = [self.selected]
            master_df['MRMR_Not_Selected'] = [self.not_selected]

        master_df.to_excel(os.path.join(path,file_name),index=False)

    def change_folder_name_complete(self):

        old_name = self.save_path
        new_name = old_name.replace('incomplete', 'complete')
        os.rename(old_name, new_name)


# models = [SvmModel, RfModel, CBModel]
#models = [SVRLinear, RandomForestEnsemble, CatBoostReg]
models = [SVR_Model, RandomForestEnsemble, CatBoostReg]
         # Neural_Network]
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
SVM_Reg = {'C': [0.0001,0.01,0.1,1,5,10,50, 100, 200,500, 1000],  # np.logspace(-6, 3,10), #*
           'gamma': np.logspace(-4, 2, 7),  # Kernel coefficient for rbf, poly & sig
           # Hashing out RBF provides more variation in the proba_values, however, the uniqueness 12 counts
           'kernel': [  'rbf',
               #'poly',
               'sigmoid',
               'linear'  # *
           ]
          #  ,'coef0': [0,0.0001,0.001,0.01,0.1, 1,10,20,100] #Independent term for poly & sig
           #, 'degree': [1,2, 3, 4] #Poly
    #, 'epsilon': [0,0.001,0.01,0.05,0.1, 0.5, 1, 5, 10, 50, 100]
  #  , 'tol': np.logspace(-4, 1, 5)
           }

SVM_Reg['gamma'] = np.append(SVM_Reg['gamma'],['auto','scale'])

RFE_Reg = {'n_estimators': [int(x) for x in np.linspace(start=200,
                                                        stop=1000,
                                                        num=9)],
            'criterion': ['squared_error', 'absolute_error', 'poisson'],
           'max_features': ['auto', 'sqrt', 'log2'],
           # Hashing out RBF provides more variation in the proba_values, however, the uniqueness 12 counts
           #'max_depth': [int(x) for x in np.linspace(1, 110,num=12)],
           'bootstrap': [False, True],
           # 'min_samples_leaf': [float(x) for x in np.arange(0.1, 0.6, 0.1)]
           }
CatBoost_Reg = {'learning_rate': [0.01,0.03, 0.1,1],
                'depth': [2,4, 6]  # DEFAULT is 6. Decrease value to prevent overfitting
    , 'l2_leaf_reg': [10,15,30,50] # Increase the value to prevent overfitting DEFAULT is 3
                }

SVRLinear = {'C': [1,70,100,200,300,1000],
            # 'tol':  np.logspace(-20, 0,21),
            # 'epsilon': [ 0,0.0001,0.1, 0.2, 0.5, 1, 5, 10, 50, 70],
            # 'loss': ['epsilon_insensitive', 'squared_epsilon_insensitive'],
            # 'fit_intercept': [True, False],
             'max_iter': [100000, 1000000,10000000]


}


######Genetic Search Opt
#from sklearn_genetic.space import Integer, Categorical, Continuous
#SVM_Reg_Genetic = {'C': Continuous(0.001,100)
           #'gamma': Continuous(-1,10,distribution="uniform") # Kernel coefficient for rbf, poly & sig
           # Hashing out RBF provides more variation in the proba_values, however, the uniqueness 12 counts
 #          ,'loss': Categorical(['epsilon_insensitive','squared_epsilon_insensitive'])

           #'kernel':Categorical(['rbf',
               #'poly',
               #'sigmoid'
            #   'linear'  # *
           #])
            #,'coef0': Integer(0, 1) #Independent term for poly & sig
           #, 'degree':Integer(1, 4) #Poly
  # , 'epsilon': Integer(0, 50)
    #, 'tol': np.logspace(-6, 1, 8)
   # ,'max_iter': Categorical([1000000])
    #       }
#RFE_Reg_genetic = {'n_estimators': Integer (100,1000),
 #                  'bootstrap': Categorical([True,False])}
NN_Regression = {
    #'initializer': ['normal', 'uniform'],
    #'activation': ['relu', 'sigmoid'],
    #'optimizer': ['adam', 'rmsprop'],
    #'loss': ['mse', 'mae'],
    'batch_size': [32, 64],
    'epochs': [5, 10],
}



grid_params = {}
grid_params['SVC'] = SvmModel
grid_params['Random_Forest'] = RfModel
grid_params['CatBoostClass'] = CBModel
#grid_params['SVR_Linear'] = SVM_Reg_Genetic
grid_params['RFE_Regressor'] = RFE_Reg
grid_params['CatBoostReg'] = CatBoost_Reg
grid_params['SVR_Linear'] = SVRLinear
grid_params['SVR'] = SVM_Reg
grid_params['NN_Reg'] = NN_Regression


save_path = regression_output_path_1
alg = Algorithm(models, select=max_std_sampling, model_type='Regression',
                scoring_type='r2', save_path=save_path, split_ratio=0.30,
                column_removal_experiment=['xlogp_cp_1','xlogp_cp_2','xlogp_cp_3',
                                           'aromatic_bond_cp_2','complexity_cp_3',
                                           'complexity_cp_2','complexity_cp_1','Final_Concentration',
                                           'final_lipid_volume','component_1_vol','component_2_vol',
                                           'component_3_vol','component_4_vol','component_1_vol_conc',
                                           'tpsa_cp_3','tpsa_cp_2',
                                           #'heavy_atom_count_cp_1','heavy_atom_count_cp_2','heavy_atom_count_cp_3',
                                           #'single_bond_cp_1','double_bond_cp_1',
                                           #'single_bond_cp_2','double_bond_cp_2',
                                           #'single_bond_cp_3','double_bond_cp_3',
                                           #'h_bond_donor_count_cp_2','h_bond_acceptor_count_cp_2',
                                           #'h_bond_donor_count_cp_3','h_bond_acceptor_count_cp_3',
                                           #'ssr_cp_2','ssr_cp_3',
                                            'Req_Weight_1', 'Ethanol_1',
                                            'Req_Weight_2', 'Ethanol_2',
                                            'Req_Weight_3',
                                            'Req_Weight_4', 'Ethanol_4'
                                            #'ethanol_dil',
                                            #'component_1_vol_stock',
                                            #'component_2_vol_stock',
                                            #'component_3_vol_stock'

                                           ],
                MRMR_K_Value=25)

#alg.analyse_data()
alg.run_algorithm(initialisation='gridsearch', splits=3, grid_params=grid_params, skip_unlabelled_analysis=True, verbose=10, kfold_repeats=2)
alg.compare_query_changes()
alg.similairty_scoring(method='gower', threshold=0.8, n_instances=10)
alg.output_data()
alg.random_unlabelled(n_instances=10)
alg.master_file()
alg.change_folder_name_complete()

