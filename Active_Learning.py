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
import re
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
from Models import SvmModel, RfModel, CBModel, SVR_Model, RandomForestEnsemble, CatBoostReg, SVRLinear#, Neural_Network
from PreProcess import MinMaxScaling, Standardisation
from QueriesCommittee import max_disagreement_sampling, max_std_sampling
from SelectionFunctions import SelectionFunction
from TrainModel import TrainModel
from confusion_matrix_custom import make_confusion_matrix
from build_al_data_file import ALDataBuild
from mrmr_algorithm import MRMR
from scipy.stats import loguniform
import warnings
from automated_liha_params import LiHa_Params
warnings.filterwarnings("ignore", category=DeprecationWarning)
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

    #Quick fix
    ul_df['ethanol_dil'] = 0.00

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

    ul_df.drop_duplicates(inplace=True)
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
                 al_folder: str,
                 random_folder: str,
                 al_random_folder: str,
                 scoring_type: str,
                 hide: str = None,
                 select: Callable = SelectionFunction.entropy_sampling,
                 file: str = 'unlabelled_data_full.csv',
                 limit: int = -1, perc_uncertain: float = 0.1, n_instances: Union[int, str] = 'Full',
                 split_ratio: float = 0.2,
                 column_removal_experiment: list=None,
                 MRMR_K_Value: int=10,
                 post_run: bool = False,
                 run_type: Optional[str]=None):

        assert perc_uncertain <= 1
        assert model_type in ['Classification', 'Regression']
        assert run_type in [None, 'AL','Random','Random_Adjusted', 'AL & Random'], 'Choices are "None", "AL", "Random" and "AL & Random" '

        self.regression = None
        self.similarity_score = None
        self.selection_probas_val = None

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
        self.post_run = post_run
        self.run_type = run_type
        self.sorted_list_of_compl_folders = None
        #######New Data - from self.create_data()/self.normalise_data()
        self.X_val = None
        self.y_train = None
        self.X_train = None
        self.y_test = None
        self.X_test = None
        self.normaliser = None
        self.count_of_iter = None
        self.prev_gguid_df = None

        self.al_folder = al_folder
        self.random_folder = random_folder
        self.al_random_folder = al_random_folder

        ####Clustering
        self.best_k = None
        self.results = None
        ##############Things to run
        self.create_save_folder()
        self.create_data()
        self.normalise_data()

    def create_save_folder(self):

        if self.post_run == False:
            dirName = str(self.uuid) +'_'+ str(self.today_date_time)+str("_iteration_")+str(0)

            if self.run_type == 'AL & Random':
                try:
                    al_random_folder = self.al_random_folder
                    save_folder = os.path.join(self.save_path,al_random_folder,dirName)
                    os.mkdir(save_folder)
                    self.save_path = save_folder
                    print("Directory ", dirName, " Created ")
                except FileExistsError:
                    print("Directory ", dirName, " already exists")
            else:
                try:
                    # Create Directory
                    save_folder = os.path.join(self.save_path, dirName)
                    os.mkdir(save_folder)
                    print("Directory ", dirName, " Created ")
                    self.save_path = save_folder
                except FileExistsError:
                    print("Directory ", dirName, " already exists")
        elif self.post_run == True:
            if self.run_type == 'AL & Random':
                al_random_folder = self.al_random_folder
                save_folder = os.path.join(self.save_path, al_random_folder)
                self.input_folder = save_folder
                list_of_folders = [subdir for root, subdir, rest in os.walk(save_folder)]
                list_of_folders = list(filter(None, list_of_folders))

                string = ''.join(str(folder) for folder in list_of_folders)
                list_of_compl_folders = re.findall(r"(?<=')(\S+_\d+_\d+_complete_iteration_\d+)", string)
                if len(list_of_compl_folders) != 0:
                    # This appears to be broken at the moment
                    # self.sorted_list_of_compl_folders = sorted(list_of_compl_folders, key=lambda  x: int("".join([i for i in x if i.isdigit()])),
                    # reverse=True)
                    self.sorted_list_of_compl_folders = sorted(list_of_compl_folders,
                                                               key=lambda x: int(re.search(r'\d+$', x).group()),
                                                               reverse=True)
                    self.count_of_iter = (len(list_of_compl_folders) - 1)
                else:
                    self.sorted_list_of_compl_folders = None
                    self.count_of_iter = 0

                dirName = str(self.uuid) + '_' + str(self.today_date_time)+str("_iteration_")+str(self.count_of_iter+1)

                try:
                    self.save_path = os.path.join(save_folder,dirName)
                    os.mkdir(self.save_path)
                    print("Directory ", dirName, " Created")
                except FileExistsError:
                    print("Directory ", dirName, " already exists")
            if self.run_type == 'AL':
                al_folder = self.al_folder
                save_folder = os.path.join(self.save_path, al_folder)
                self.input_folder = save_folder
                list_of_folders = [subdir for root, subdir,rest in os.walk(save_folder)]
                list_of_folders =list(filter(None, list_of_folders))

                string = ''.join(str(folder) for folder in list_of_folders)
                list_of_compl_folders = re.findall(r"(?<=')(\S+_\d+_\d+_complete_iteration_\d+)", string)

                if len(list_of_compl_folders) != 0:
                    #This appears to be broken at the moment
                    #self.sorted_list_of_compl_folders = sorted(list_of_compl_folders, key=lambda  x: int("".join([i for i in x if i.isdigit()])),
                                                      #reverse=True)
                    self.sorted_list_of_compl_folders = sorted(list_of_compl_folders,
                                                               key=lambda x: int(re.search(r'\d+$',x).group()),
                                                               reverse=True)
                    self.count_of_iter = (len(list_of_compl_folders) - 1)
                else:
                    self.sorted_list_of_compl_folders = None
                    self.count_of_iter = 0



                dirName = str(self.uuid) + '_' + str(self.today_date_time)+str("_iteration_")+str(self.count_of_iter+1)

                try:
                    self.save_path = os.path.join(save_folder,dirName)
                    os.mkdir(self.save_path)
                    print("Directory ", dirName, " Created")
                except FileExistsError:
                    print("Directory ", dirName, " already exists")
            elif self.run_type == 'Random':
                rand_folder = self.random_folder
                save_folder = os.path.join(self.save_path,rand_folder)
                self.input_folder = save_folder
                list_of_folders = [subdir for root, subdir,rest in os.walk(save_folder)]
                list_of_folders =list(filter(None, list_of_folders))

                string = ''.join(str(folder) for folder in list_of_folders)
                list_of_compl_folders = re.findall(r"(?<=')(\S+_\d+_\d+_complete_iteration_\d+)", string)
                if len(list_of_compl_folders) != 0:
                    #This appears to be broken at the moment
                    #self.sorted_list_of_compl_folders = sorted(list_of_compl_folders, key=lambda  x: int("".join([i for i in x if i.isdigit()])),
                                                      #reverse=True)
                    self.sorted_list_of_compl_folders = sorted(list_of_compl_folders,
                                                               key=lambda x: int(re.search(r'\d+$',x).group()),
                                                               reverse=True)
                    self.count_of_iter = (len(list_of_compl_folders) - 1)
                else:
                    self.sorted_list_of_compl_folders = None
                    self.count_of_iter = 0



                dirName = str(self.uuid) + '_' + str(self.today_date_time)+str("_iteration_")+str(self.count_of_iter+1)
                try:
                    self.save_path = os.path.join(save_folder, dirName)
                    os.mkdir(self.save_path)
                    print("Directory ", dirName, " Created")
                except FileExistsError:
                    print("Directory ", dirName, " already exists")
            elif self.run_type == 'Random_Adjusted':
                rand_folder = self.random_folder
                save_folder = os.path.join(self.save_path,rand_folder)
                self.input_folder = save_folder
                list_of_folders = [subdir for root, subdir,rest in os.walk(save_folder)]
                list_of_folders =list(filter(None, list_of_folders))

                string = ''.join(str(folder) for folder in list_of_folders)
                list_of_compl_folders = re.findall(r"(?<=')(\S+_\d+_\d+_complete_iteration_\d+)", string)
                if len(list_of_compl_folders) != 0:
                    #This appears to be broken at the moment
                    #self.sorted_list_of_compl_folders = sorted(list_of_compl_folders, key=lambda  x: int("".join([i for i in x if i.isdigit()])),
                                                      #reverse=True)
                    self.sorted_list_of_compl_folders = sorted(list_of_compl_folders,
                                                               key=lambda x: int(re.search(r'\d+$',x).group()),
                                                               reverse=True)
                    self.count_of_iter = (len(list_of_compl_folders) - 1)
                else:
                    self.sorted_list_of_compl_folders = None
                    self.count_of_iter = 0



                dirName = str(self.uuid) + '_' + str(self.today_date_time)+str("_iteration_")+str(self.count_of_iter+1)
                try:
                    self.save_path = os.path.join(save_folder, dirName)
                    os.mkdir(self.save_path)
                    print("Directory ", dirName, " Created")
                except FileExistsError:
                    print("Directory ", dirName, " already exists")
    def create_data(self):
        '''
        The create data function will do various things. Firstly, it will run the unlabelled data function that creates
        the unlabelled data file in the correct format.

        Then, Data Load Split class will be instantiated. The calling of this class will read in the initial results of
        the experiments (prior to any labelling of unlabelled data), and set up the X and y arrays in the format req.
        for the algorithms.
        The Data Load Split also contains an analysis function that will produce graphs of the X and y data. It will also
        determine whether the y data is gaussian or not.

        :return:
        '''

        # Pull in unlabelled Data




        # uncertain_count = math.ceil(len(self.X_val) * self.perc_uncertain)

        # Run Split
        #DLS onlly accurate up to 1000.0 size, need to filter

        data_load_split = Data_Load_Split("Results_Complete.csv", hide_component=self.hide, alg_categ=self.model_type,
                                          split_ratio=self.split_ratio,
                                          shuffle_data=True, filter_target=True, target='Z-Average (d.nm)',
                                          smaller_than=1000.0, column_removal_experiment=self.column_removal_experiment)
        self.converted_columns = data_load_split.columns_converted
        self.target_labels = data_load_split.class_names_str




        self.X_train, self.X_test, self.y_train, self.y_test = data_load_split.split_train_test()

        if self.post_run==True:
            if self.run_type == 'AL':
                folder_formulations = r'/Users/calvin/Library/CloudStorage/OneDrive-Personal/Documents/2022/RegressorCommittee_Output/'
                folder_formulations = os.path.join(folder_formulations,self.al_folder)
            elif self.run_type == 'Random':
                folder_formulations = r'/Users/calvin/Library/CloudStorage/OneDrive-Personal/Documents/2022/RegressorCommittee_Output/'
                folder_formulations = os.path.join(folder_formulations,self.random_folder)
            elif self.run_type == 'Random_Adjusted':
                folder_formulations = r'/Users/calvin/Library/CloudStorage/OneDrive-Personal/Documents/2022/RegressorCommittee_Output/'
                folder_formulations = os.path.join(folder_formulations,self.random_folder)
            elif self.run_type == 'AL & Random':
                folder_formulations = r'/Users/calvin/Library/CloudStorage/OneDrive-Personal/Documents/2022/RegressorCommittee_Output/'
                folder_formulations = os.path.join(folder_formulations, self.al_random_folder)
            more_data = ALDataBuild(folder_dls=r'/Users/calvin/Library/CloudStorage/OneDrive-Personal/Documents/2022/RegressorCommittee_Input',
                                    folder_formulations=folder_formulations)
            #TODO Check out the sorted list of compl folders issue
            self.prev_gguid_df = more_data.load_prev_gguid(save_path=self.input_folder,recent_folder=self.sorted_list_of_compl_folders[0])
            if self.run_type != 'Random_Adjusted':
                more_data.collect_csv()
                more_data.clean_dls_data()
                more_data.filter_out_D_iteration()
                more_data.z_scoring(threshold=1.0)
                more_data.collect_formulations()
                more_data.merge_dls_data()
            if self.sorted_list_of_compl_folders is None:
                self.X_val, self.columns_x_val = unlabelled_data(self.file,
                                                                 method='fillna',
                                                                 column_removal_experiment=self.column_removal_experiment)  # TODO need to rename
                self.X_train_prev, self.y_train_prev = None, None
            else:
                self.X_train_prev, self.y_train_prev = more_data.load_x_y_prev_run(save_path =self.input_folder
                                                                                   ,recent_folder=self.sorted_list_of_compl_folders[0])
                self.X_val, self.columns_x_val, self.x_val_prev_index = more_data.load_x_val_prev_run(save_path =self.input_folder ,
                                                                               recent_folder= self.sorted_list_of_compl_folders[0])

            #temporary solution
            #TODO Need to fix this
            if self.run_type == 'AL & Random':
                self.X_val, self.columns_x_val = more_data.remove_AL_from_unlabelled(X_val=self.X_val,
                                                                                     X_val_columns_names=self.columns_x_val,
                                                                                     x_prev_index=self.x_val_prev_index,
                                                                                     count=(self.count_of_iter+1))

                self.X_train_AL, self.y_train_AL = more_data.return_AL_data()
                self.X_train, self.y_train = more_data.add_AL_to_train(X_train_initial=self.X_train, y_train_initial=self.y_train,
                            X_train_prev=self.X_train_prev, y_train_prev=self.y_train_prev)
                data_load_split.update_x_y_data(additional_x=more_data.X_train_AL,additional_y=more_data.y_train_AL,
                                                prev_x_data=self.X_train_prev, prev_y_data=self.y_train_prev)
            if self.run_type == 'AL':
                self.X_val, self.columns_x_val = more_data.remove_AL_from_unlabelled(X_val=self.X_val,
                                                                                     X_val_columns_names=self.columns_x_val,
                                                                                     x_prev_index=self.x_val_prev_index,
                                                                                     count= (self.count_of_iter+1))
                self.X_train_AL, self.y_train_AL = more_data.return_AL_data()
                self.X_train, self.y_train = more_data.add_AL_to_train(X_train_initial=self.X_train, y_train_initial=self.y_train,
                            X_train_prev=self.X_train_prev, y_train_prev=self.y_train_prev)

                #This updates the main X and y arrays ion the Data Load Split so that when new data is added, the analysis
                #can be done on ALSO the new data
                data_load_split.update_x_y_data(additional_x=more_data.X_train_AL,additional_y=more_data.y_train_AL,
                                                prev_x_data=self.X_train_prev, prev_y_data=self.y_train_prev)

            elif self.run_type == 'Random':




                self.X_val, self.columns_x_val = more_data.remove_random_from_unlabelled(X_val=self.X_val,
                                                                                     X_val_columns_names=self.columns_x_val,
                                                                                     x_prev_index=self.x_val_prev_index,
                                                                                     count= (self.count_of_iter+1))
                self.X_train_random, self.y_train_random = more_data.return_random_data()
                self.X_train, self.y_train = more_data.add_random_to_train(X_train_initial=self.X_train, y_train_initial=self.y_train,
                            X_train_prev=self.X_train_prev, y_train_prev=self.y_train_prev)


                data_load_split.update_x_y_data(additional_x=more_data.X_train_random, additional_y=more_data.y_train_random,
                                                prev_x_data=self.X_train_prev,prev_y_data= self.y_train_prev)

            elif self.run_type == 'Random_Adjusted':
                self.X_val, self.columns_x_val = unlabelled_data(self.file,
                                                                 method='fillna',
                                                                 column_removal_experiment=self.column_removal_experiment)
                self.X_train_random, self.y_train_random = more_data.return_random_adjusted(column_selection=self.columns_x_val,save_path=self.input_folder, folder=self.sorted_list_of_compl_folders[0])
                self.X_train, self.y_train = more_data.add_random_adjusted_to_train(X_train_initial=self.X_train, y_train_initial=self.y_train,
                            X_train_prev=self.X_train_prev, y_train_prev=self.y_train_prev)
                data_load_split.update_x_y_data(additional_x=self.X_train_random, additional_y=self.y_train_random,
                                                prev_x_data=self.X_train_prev,prev_y_data= self.y_train_prev)


            elif self.run_type is None:
                pass

        elif self.post_run==False:
            self.X_val, self.columns_x_val = unlabelled_data(self.file,
                                                             method='fillna',
                                                             column_removal_experiment=self.column_removal_experiment)  # TODO need to rename

            self.X_train_prev, self.y_train_prev = None, None



        mrmr = MRMR(data_load_split.X, data_load_split.y, self.columns_x_val, K=self.K)
        self.selected, self.not_selected = mrmr.computing_correlations()

        self.gaussian_result = data_load_split.analyse_data(save_path=self.save_path, column_names=self.columns_x_val,
                                                            plot=False)

        ### Log Transform y target
        #TODO Implement better

        if self.gaussian_result == False:
            self.y_train = np.log(self.y_train)
            self.y_test = np.log(self.y_test)
        else:
            pass
        ###

        random.seed(42) #This shhould ensure shuffling is always the same
        #shuffle seems to change values...
        #np.random.shuffle() may be better.
        #But why bother shuffling...
        #https: // stackoverflow.com / questions / 44917606 / python - why - does - random - shuffle - change - the - array
        # if self.limit > 0:
        #     shuffle(self.X_val)
        #     self.X_val = self.X_val[:self.limit]
        # else:
        #     np.random.shuffle(self.X_val)

        if isinstance(self.n_instances, int):
            self.n_instances = self.n_instances
        elif isinstance(self.n_instances, str):
            if self.n_instances in ['Full', 'full', 'All', 'all']:
                self.n_instances = len(self.X_val)
            elif self.n_instances in ['Half', 'half']:
                self.n_instances = round(len(self.X_val) / 2)




        test = pd.DataFrame(self.X_val, columns=self.columns_x_val)

        print(test.info())
        print(test['mw_cp_2'].value_counts())

        self.X = data_load_split.X
        self.y = data_load_split.y

        return self.X_train, self.X_test, self.y_train, self.y_test

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
            if initialisation in ['gridsearch','randomized'] :
                self.scores = self.committee_models.gridsearch_committee(initialisation=initialisation, grid_params=grid_params, verbose=verbose)
            if initialisation =='default':
                self.scores = self.committee_models.default_committee()
            elif initialisation == 'optimised':
                self.scores = self.committee_models.optimised_comittee(params=grid_params)
            self.committee_models.fit_data()
            self.score_data = self.committee_models.score()
            self.rmse_score_data = self.committee_models.rmse_scoring()
            self.selection_probas_val, *rest = self.committee_models.query(self.committee_models,
                                                                           n_instances=self.n_instances)
            # test = batch_sampling(models=self.committee_models, X=self.X_val, X_labelled=self.X_train,
            #                      converted_columns=self.converted_columns, query_type=max_std_sampling, n_jobs=-1,
            #                      metric='gower')
            self.committee_models.predictionvsactual(save_path=self.save_path, plot=False)
            self.models_algorithms = self.committee_models.printname()
            self.committee_models.out_cv_score(save_path=self.save_path)
            #self.committee_models.shap_analysis_committee(X_test=self.X_test, X=self.X ,features=self.columns_x_val,
            #                                              y_test=self.y_test,
            #                                              save_path=self.save_path)
            self.committee_models.shapash_analysis_committee(X_train=self.X_train, y_train=self.y_train,
                                                             X_test=self.X_test, X=self.X, features=self.columns_x_val,
                                                             y_test=self.y_test,y= self.y)


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

        self.df = pd.DataFrame(reversed_x_val, columns=self.columns_x_val, index=self.samples_index)
        # add similarty scores:
        self.sample_score.index = self.df.index
        self.df['sample_scoring'] = self.sample_score
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

        self.output_file_name = str(self.models_algorithms) + '_Ouput_Selection_' + str(self.select.__name__) + '_' + str(
            self.today_date) + '.xlsx'
        file_name = os.path.join(self.save_path, self.output_file_name)

        writer = pd.ExcelWriter(file_name
                                ,
                                engine='xlsxwriter')
        df_info.to_excel(writer,
                         index=False, startrow=1)
        df_scores.to_excel(writer,
                           index=False, startcol=col_move)

        self.df.to_excel(
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
            df_scores_data_rmse = pd.DataFrame.from_dict(self.rmse_score_data, orient='index').T
            df_scores_data = pd.concat([df_scores_data, df_scores_data_rmse])
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
        if self.run_type != 'AL & Random':
            random_data = unlabeled_data.sample(n=n_instances, random_state=42)
            file_name = "random_unlabeled_data_points.xlsx"
            file_name = os.path.join(self.save_path, file_name)
            random_data.to_excel(file_name, index_label=False)
        else:
            temp_df = self.df.index.to_numpy().reshape(-1,1)
            temp_df = pd.DataFrame(temp_df, columns=['original_index'])
            original_shape = unlabeled_data.shape
            unlabeled_data = unlabeled_data.merge(temp_df, left_on='original_index', right_on='original_index', how='left', indicator=True)
            unlabeled_data = unlabeled_data[unlabeled_data['_merge'] == 'left_only']
            merge_shape = unlabeled_data.shape
            both_df = unlabeled_data[unlabeled_data['_merge'] == 'both']
            print("Original Shape: ", original_shape)
            print("Merge Shape: ", merge_shape)
            random_data = unlabeled_data.sample(n=n_instances, random_state=42)
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
            df_temp['run_type'] = self.run_type
            df_temp['iteration'] = self.count_of_iter+1 if self.count_of_iter is not None else 0

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

    def export_modified_unlabelled_data_and_additional_labeled_and_guid(self):


        _, reversed_x_val, _ = self.normaliser.inverse(self.X_train, self.X_val, self.X_test,
                                                       converted_columns=self.converted_columns)
        unlabeled_data= pd.DataFrame(reversed_x_val, columns=self.columns_x_val)
        unlabeled_data['original_index'] = self.selection_probas_val[0]
        unlabeled_data.loc[unlabeled_data.Ratio_2 == 0.44999999999999996] = 0.45
        print(unlabeled_data.info())
        print(unlabeled_data['mw_cp_2'].unique())
        print(unlabeled_data['mw_cp_2'].value_counts())
        unlabeled_data.to_csv(os.path.join(self.save_path, "Unlabeled_Data.csv"), index=True)

        #Need
        if self.post_run == True:
            if self.X_train_prev is not None and self.y_train_prev is not None:
                if self.run_type == 'AL':
                    add_data_tgthr_x = np.vstack((self.X_train_prev, self.X_train_AL))
                    add_data_tgthr_y = np.hstack((self.y_train_prev, self.y_train_AL)).reshape(-1,1)
                    added_data = pd.DataFrame(add_data_tgthr_x, columns=self.columns_x_val)
                    added_data['Z-Average (d.nm)'] = add_data_tgthr_y
                    added_data.to_csv(os.path.join(self.save_path, "Added_Data.csv"), index = False)
                elif self.run_type == 'Random':
                    add_data_tgthr_x = np.vstack((self.X_train_prev, self.X_train_random))
                    add_data_tgthr_y = np.hstack((self.y_train_prev, self.y_train_random)).reshape(-1, 1)
                    added_data = pd.DataFrame(add_data_tgthr_x, columns=self.columns_x_val)
                    added_data['Z-Average (d.nm)'] = add_data_tgthr_y
                    added_data.to_csv(os.path.join(self.save_path, "Added_Data.csv"), index=False)
                elif self.run_type == 'Random_Adjusted':
                    add_data_tgthr_x = np.vstack((self.X_train_prev, self.X_train_random))
                    add_data_tgthr_y = np.hstack((self.y_train_prev, self.y_train_random)).reshape(-1, 1)
                    added_data = pd.DataFrame(add_data_tgthr_x, columns=self.columns_x_val)
                    added_data['Z-Average (d.nm)'] = add_data_tgthr_y
                    added_data.to_csv(os.path.join(self.save_path, "Added_Data.csv"), index=False)
                elif self.run_type == 'AL & Random':
                    add_data_tgthr_x = np.vstack((self.X_train_prev, self.X_train_AL))
                    add_data_tgthr_y = np.hstack((self.y_train_prev, self.y_train_AL)).reshape(-1,1)
                    added_data = pd.DataFrame(add_data_tgthr_x, columns=self.columns_x_val)
                    added_data['Z-Average (d.nm)'] = add_data_tgthr_y
                    added_data.to_csv(os.path.join(self.save_path, "Added_Data.csv"), index=False)
            else:
                print("No previous iteration data")
                if self.run_type == 'AL':
                    added_data = pd.DataFrame(self.X_train_AL, columns=self.columns_x_val)
                    added_data['Z-Average (d.nm)'] = self.y_train_AL.reshape(-1, 1)
                    added_data.to_csv(os.path.join(self.save_path, "Added_Data.csv"), index=False)
                elif self.run_type == 'Random':
                    added_data = pd.DataFrame(self.X_train_random, columns=self.columns_x_val)
                    added_data['Z-Average (d.nm)'] = self.y_train_random.reshape(-1, 1)
                    added_data.to_csv(os.path.join(self.save_path, "Added_Data.csv"), index=False)

                elif self.run_type == 'Random_Adjusted':
                    added_data = pd.DataFrame(self.X_train_random, columns=self.columns_x_val)
                    added_data['Z-Average (d.nm)'] = self.y_train_random.reshape(-1, 1)
                    added_data.to_csv(os.path.join(self.save_path, "Added_Data.csv"), index=False)
                if self.run_type == 'AL & Random':
                    added_data = pd.DataFrame(self.X_train_AL, columns=self.columns_x_val)
                    added_data['Z-Average (d.nm)'] = self.y_train_AL.reshape(-1, 1)
                    added_data.to_csv(os.path.join(self.save_path, "Added_Data.csv"), index=False)

            temp_new_guid = pd.DataFrame([np.array((self.count_of_iter+1, self.uuid))],columns=['iteration', 'guid'])
        elif self.post_run == False:
            print("First run - No iteration data")
            temp_new_guid = pd.DataFrame([np.array(((0), self.uuid))], columns=['iteration', 'guid'])
        if self.prev_gguid_df is not None:
            df2 = pd.concat([self.prev_gguid_df, temp_new_guid]).reset_index(drop=True)
        else:
            df2 = temp_new_guid
        df2.to_csv(os.path.join(self.save_path,'iteration_prev_guid.csv'),index=False)


    def automated_liha_file(self):

        liha = LiHa_Params(self.output_file_name,
                   self.save_path)

        liha.read_al_file()
        liha.reactant_weights(0.5, 0.5, 0.3)


    def change_folder_name_complete(self):

        old_name = self.save_path
        new_name = old_name.replace('incomplete', 'complete')
        os.rename(old_name, new_name)


# models = [SvmModel, RfModel, CBModel]
#models = [SVRLinear, RandomForestEnsemble, CatBoostReg]
models = [SVR_Model, RandomForestEnsemble, CatBoostReg]
         # Neural_Network]
# selection_functions = [selection_functions]

#SvmModel = {'C': np.logspace(-3, 3, 8),
#            'gamma': np.logspace(-5, 3, 9),
            # Hashing out RBF provides more variation in the proba_values, however, the uniqueness 12 counts
#            'kernel': ['rbf', 'poly', 'sigmoid', 'linear'],
#            'coef0': [0, 0.001, 0.1, 1],
 #           'degree': [1, 2, 3, 4]}

RfModel = {'n_estimators': [int(x) for x in np.linspace(start=200, stop=2000, num=10)],
                             'max_features': ['auto', 'sqrt'],
                             'max_depth': [int(x) for x in np.linspace(10, 35, num=11)]
            ,'min_samples_split': [2, 5,7,8,9, 10]
            ,'min_samples_leaf': [1, 2,3, 4,5]
            ,'bootstrap': [True, False]
           }

CBModel = {
    'depth': [3, 1, 2, 6, 4, 5, 7, 8, 9, 10]
    #    , 'n_estimators': [250, 100, 500, 1000]
    #    , 'learning_rate': [0.03, 0.001, 0.01, 0.1, 0.2, 0.3]
    #    , 'l2_leaf_reg': [3, 1, 5, 10, 100],
    #    'border_count': [32, 5, 10, 20, 50, 100, 200]
}
# https://www.vebuso.com/2020/03/svm-hyperparameter-tuning-using-gridsearchcv/
SVM_Reg = {'C': np.logspace(-4, 3,8),
           #'gamma': np.logspace(-9, 3, 13), #np.logspace(-4, 2, 7),  # Kernel coefficient for rbf, poly & sig
           # Hashing out RBF provides more variation in the proba_values, however, the uniqueness 12 counts
           'kernel': [  'rbf',
           #    'poly',
               'sigmoid',
               'linear'  # *
           ]
           #,'coef0': [0,0.0001,0.001,0.01,0.1, 1,10,20,100] #Independent term for poly & sig
           #, 'degree': [1,2, 3, 4] #Poly
    , 'epsilon': np.logspace(-3,1,5) #http://adrem.uantwerpen.be/bibrem/pubs/IJCNN2007.pdf https://stackoverflow.com/questions/69669827/tuning-of-hyperparameters-of-svr
    #, 'tol': np.logspace(-4, 1, 6)
           }






max_depth = [int(x) for x in np.linspace(1, 110,num=12)]
max_depth.append(None)



RFE_Reg = {'n_estimators': [int(x) for x in np.linspace(start=200,
                                                        stop=2000,
                                                        num=10)],
            'criterion': ['squared_error'],
           'max_features': [None, 'sqrt', 'log2'],
           'max_depth': max_depth,
           'bootstrap': [False, True],
            'min_samples_leaf': [float(x) for x in np.arange(0.1, 0.6, 0.1)],
           'min_samples_split': [2,5,10]
           }
#RFE_Reg = {'n_estimators': 1000, 'max_features':'sqrt'}


CatBoost_Reg = {'learning_rate': np.logspace(-4,2,7),
                'depth': [1,2,4,5, 6]  # DEFAULT is 6. Decrease value to prevent overfitting
    , 'l2_leaf_reg': [3,5,9,10,15,30,50] # Increase the value to prevent overfitting DEFAULT is 3
                }
#CatBoost_Reg = {'depth':2,'l2_leaf_reg':15,'learning_rate':0.1}
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



# ##New Params 7buq7aqpdy - guid
# SVM_Reg = {'C': 50,
#            'kernel': 'linear'
#           ,'coef0': 100
#            }
# RFE_Reg = {'n_estimators': 700
#             ,'criterion': 'absolute_error',
#            'max_features': 'sqrt',
#            # Hashing out RBF provides more variation in the proba_values, however, the uniqueness 12 counts
#            'max_depth': 100,
#            'bootstrap': False,
#             'min_samples_leaf': 0.1
#            }
#
# CatBoost_Reg = {'learning_rate': 0.5,
#                 'depth': 2  # DEFAULT is 6. Decrease value to prevent overfitting
#     , 'l2_leaf_reg': 10 # Increase the value to prevent overfitting DEFAULT is 3
#                 }



###Reduced Features - New Params
# SVM_Reg = {'C': 10
#          ,'epsilon': 0.001
#           }
#
# RFE_Reg = {'n_estimators': 800,

#
# RFE_Reg = {'n_estimators': 600,
#             'max_features': 'sqrt',
#             'max_depth': 100,
#             'bootstrap': False,
#            'min_samples_split': 10,
#              'min_samples_leaf': 0.1
#             }
# CatBoost_Reg = {'learning_rate': 0.1,
#                  'depth': 6  # DEFAULT is 6. Decrease value to prevent overfitting
#      , 'l2_leaf_reg': 5 # Increase the value to prevent overfitting DEFAULT is 3
#                 ,'verbose': False
#                  }
#
#
#
# #Default RFE
# RFE_Reg = {'n_estimators' : 100,
#            'criterion': 'squared_error',
#            'max_depth': None,
#            'min_samples_split': 2,
#            'min_samples_leaf': 1,
#            'min_weight_fraction_leaf': 0.0,
#            'max_features': 1.0,
#            'max_leaf_nodes': None,
#           'min_impurity_decrease': 0.0,
#            'bootstrap': True,
#            'n_jobs': -1,
#            'max_samples': None}
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
                scoring_type='r2', save_path=save_path, al_folder='AL_Output',
                al_random_folder='AL_Random_Output',
                random_folder='Random_Output',

                split_ratio=0.30,
                column_removal_experiment=[#'xlogp_cp_1','xlogp_cp_2',
                                                                     'xlogp_cp_3',
                                           #'aromatic_bond_cp_2',
                                           'complexity_cp_3',
                                           #'complexity_cp_2','complexity_cp_1',
                                           'Final_Concentration',
                                           'final_lipid_volume','component_1_vol','component_2_vol','component_2_vol_conc',
                                           'component_3_vol','component_4_vol','component_1_vol_conc',
                                           'component_3_vol_conc',
                                           'tpsa_cp_3',
                                           #'tpsa_cp_2',
                                           'Ratio_3','Req_Weight_3','Overall_Concentration_3','mw_cp_3',
                                           #'heavy_atom_count_cp_1','heavy_atom_count_cp_2',
                                           'heavy_atom_count_cp_3',
                                           #'single_bond_cp_1','double_bond_cp_1',
                                           #'single_bond_cp_2','double_bond_cp_2',
                                           'single_bond_cp_3','double_bond_cp_3',
                                           'h_bond_donor_count_cp_2',
                    # 'h_bond_acceptor_count_cp_2',
                                           'h_bond_donor_count_cp_3','h_bond_acceptor_count_cp_3',
                                           #'ssr_cp_2',
                                           'ssr_cp_3',
                                            'Req_Weight_1', 'Ethanol_1',
                                            'Req_Weight_2', 'Ethanol_2',
                                            'Req_Weight_3',
                                            'Req_Weight_4', 'Ethanol_4',
                                            'ethanol_dil',
                                            'component_1_vol_stock',
                                            'component_2_vol_stock',
                                            'component_3_vol_stock'

                                           ],
                MRMR_K_Value=10
                ,post_run=False,
                run_type='AL')

#alg.analyse_data()
alg.run_algorithm(initialisation='default', splits=10, grid_params=grid_params, skip_unlabelled_analysis=True, verbose=False, kfold_repeats=10,
                  )
alg.compare_query_changes()
alg.similairty_scoring(method='gower', threshold=0.8, n_instances=5)
alg.output_data()
alg.random_unlabelled(n_instances=10)
alg.master_file()
alg.export_modified_unlabelled_data_and_additional_labeled_and_guid()
alg.automated_liha_file()
alg.change_folder_name_complete()

#https://medium.com/analytics-vidhya/mae-mse-rmse-coefficient-of-determination-adjusted-r-squared-which-metric-is-better-cd0326a5697e
#https://towardsdatascience.com/m rmr-explained-exactly-how-you-wished-someone-explained-to-you-9cf4ed27458b