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
from random import shuffle
from typing import Callable, Union, Optional

import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.metrics import *
from sqlalchemy.orm.loading import instances

import Similarity_Measure
from BaseComittee import BaseLearner
from BatchMode import uncertainty_batch_sampling
from Cluster import KMeans_Cluster, HDBScan
from CommitteeClass import CommitteeRegressor, CommitteeClassification
from DataLoad import Data_Load_Split
from DiversitySampling_Clustering import DiversitySampling
from Models import SvmModel, RfModel, RandomForestEnsemble, SVR_Model, CBModel, CatBoostReg
from QueriesCommittee import KLMaxDisagreement, max_disagreement_sampling, vote_entropy, vote_entropy_sampling, \
    max_std_sampling
from QueryInstanceDensityWeighted import QueryInstanceDensityWeighted
from SelectionFunctions import SelectionFunction
from sklearn.metrics import confusion_matrix
from PreProcess import MinMaxScaling
from TrainModel import TrainModel
import scipy.sparse as sp
from DensityWeightedUncertaintySampling import DensityWeightedUncertaintySampling, DensityWeightedLogisticRegression
from confusion_matrix_custom import make_confusion_matrix

"""'

GATHER DATA

'"""

if platform.system() =='Windows':
    os.chdir()
elif platform.system() == 'Darwin':
    wrk_path_3 = r"/Users/calvin/Documents/OneDrive/Documents/2020/Liposomes Vitamins/LiposomeFormulation"
    os.chdir(wrk_path_3)

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
    ul_df = ul_df.drop(columns=column_drop)
    ul_df.replace(np.nan, 'None', inplace=True)
    ul_df = pd.get_dummies(ul_df, columns=["Component_1", "Component_2", "Component_3"],
                           prefix="", prefix_sep="")
    # if method=='fillna':    ul_df['Component_3'] = ul_df['Component_3'].apply(lambda x: None if pd.isnull(x) else x) #TODO This should be transformed into an IF function, thus when the function for unlabelled is filled with a parameter, then activates

    ul_df = ul_df.groupby(level=0, axis=1, sort=False).sum()

    print(ul_df.isna().any())
    X_val = ul_df.to_numpy()
    columns_x_val = ul_df.columns
    return X_val, columns_x_val


'''Split the Data'''

'''

Selection

'''


class Algorithm(object):
    accuracies = []

    def __init__(self, model_object, selection_functions, model_type, hide: str = None,
                 select: Callable = SelectionFunction.entropy_sampling,
                 file: str = 'unlabelled_data_full.csv',
                 limit: int = -1, perc_uncertain: float = 0.1, n_instances: Union[int, str] = 'Full',
                 split_ratio: float = 0.2):
        self.regression = None
        self.similarity_score = None
        self.selection_probas_val = None
        assert perc_uncertain <= 1
        assert model_type in ['Classification', 'Regression']

        self.model_test = None
        self.optimised_classifier = None
        self.model_object = model_object
        self.selection_functions = selection_functions
        self.diversity_sampling = DiversitySampling()
        self.select = select
        self.file = file
        self.limit = limit
        self.n_instances = n_instances
        self.split_ratio = split_ratio

        self.perc_uncertain = perc_uncertain
        self.models_algorithms = None
        self.model_type = model_type
        self.hide = hide
        self.classification = None

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

        self.create_data()
        self.normalise_data()

    def create_data(self, select: Callable = SelectionFunction.entropy_sampling, instances: int = 1889244,
                    randomize_tie_break: bool = True,
                    hide: str = None, max_epochs: int = 10,
                    num_clusters: int = 5):

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

        uncertain_count = math.ceil(len(self.X_val) * self.perc_uncertain)

        # Run Split

        data_load_split = Data_Load_Split("Results_Complete.csv", hide_component=self.hide, alg_categ=self.model_type,
                                          split_ratio=self.split_ratio,
                                          shuffle_data=True)

        self.X_train, self.X_test, self.y_train, self.y_test = data_load_split.split_train_test()
        return self.X_train, self.X_test, self.y_train, self.y_test
        # normalise data - Increased accuracy from 50% to 89%

    def normalise_data(self):
        self.normaliser = MinMaxScaling()
        self.normaliser.minmaxscale_fit(self.X_train, self.X_test, self.X_val)
        self.X_train, self.X_val, self.X_test = self.normaliser.minmaxscale_trans(self.X_train, self.X_val, self.X_test)
        return self.X_train, self.X_val, self.X_test

    def run_algorithm(self):
        if self.model_type == 'Regression':
            self.regression_model()
            self.regression = 1
        elif self.model_type == 'Classification':
            self.classification_model()
            self.classification = 1

    def regression_model(self):
        if type(self.model_object) is list:
            self.committee_models = CommitteeRegressor(self.model_object, self.X_train, self.X_test, self.y_train,
                                                       self.y_test, self.X_val,
                                                       splits=3,
                                                       kfold_shuffle=True,
                                                       scoring_type='r2',
                                                       instances=self.n_instances,
                                                       query_strategy=max_std_sampling)
            self.scores = self.committee_models.gridsearch_committee()
            self.committee_models.fit_data()
            self.score_data = self.committee_models.score()
            self.selection_probas_val = self.committee_models.query(self.committee_models, n_instances=self.n_instances)
            self.models_algorithms = self.committee_models.printname()
            
    def classification_model(self):
        if type(self.model_object) is list:
            self.committee_models = CommitteeClassification(learner_list=self.model_object, X_training=self.X_train,
                                                            X_testing=self.X_test,
                                                            y_training=self.y_train, y_testing=self.y_test,
                                                            X_unlabeled=self.X_val,
                                                            query_strategy=max_disagreement_sampling,
                                                            c_weight='balanced',
                                                            splits=3,
                                                            scoring_type='precision',
                                                            kfold_shuffle=True)
            self.scores = self.committee_models.gridsearch_committee()
            self.committee_models.fit_data()
            # self.probas_val = self.committee_models.vote_proba()

            self.selection_probas_val = self.committee_models.query(self.committee_models, n_instances=self.n_instances,
                                                                    X_labelled=self.X_train)

            self.conf_matrix = self.committee_models.confusion_matrix()
            self.precision_scores = self.committee_models.precision_scoring()
            self.models_algorithms = self.committee_models.printname()

    def compare_query_changes(self):
        selection_df = pd.DataFrame(self.selection_probas_val[1]).reset_index(drop=True)
        unlabelled_df_temp = pd.DataFrame(self.X_val.copy()).reset_index(drop=True)
        print(selection_df.equals(unlabelled_df_temp))

    def similairty_scoring(self, method: str = 'gower', threshold: float = 0.5,n_instances: int = 100, k_range: Optional[int] = 10,
                           alpha: Optional[float] = 0.01):
        self.density_method = method
        self.method_threshold = threshold
        sim_init = Similarity_Measure.Similarity(self.selection_probas_val[1], self.selection_probas_val[0])

        if method == 'cosine':
            assert threshold <= 1.0
            self.samples, self.samples_index, self.sample_score = sim_init.similarity_cosine(threshold, 'cosine')
        elif method == 'gower':
            assert threshold <= 1.0
            self.samples, self.samples_index, self.sample_score = sim_init.similarity_gower(threshold, n_instances=n_instances)
        elif method == 'kmeans':
            self.kmeans_cluster_deopt = KMeans_Cluster(unlabeled_data=self.selection_probas_val)
            # self.kmeans_cluster.elbow_method(clusters=20)

            self.best_k, self.results = self.kmeans_cluster_deopt.chooseBestKforKmeansParallel(k_range=k_range,
                                                                                               alpha=alpha)

            self.kmeans_cluster_opt = KMeans_Cluster(unlabeled_data=self.selection_probas_val, n_clusters=self.best_k)
            self.kmeans_cluster_opt.kmeans_fit()
            self.samples, self.samples_index, self.sample_score = self.kmeans_cluster_opt.create_array(threshold=threshold,
                                                                                    n_instances=n_instances)
            # self.kmeans_cluster_deopt.silhouette(X=self.X_val)

        elif method == 'hdbscan':
            self.hdbscan_opt = HDBScan(unlabeled_data=self.selection_probas_val)
            self.hdbscan_opt.hdbscan_fit()

            self.samples, self.samples_index, self.sample_score = self.hdbscan_opt.distance_sort()



        return self.samples, self.samples_index, self.sample_score



    def single_model(self):

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

    # print('val predicted:',
    #      self.model_test.val_y_predicted.shape,
    #      self.model_test.val_y_predicted)

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

        # TODO Fix up this code completely - Choosing the right key needs to be automatic, not a guess
        _, reversed_x_val, _ = self.normaliser.inverse_minmaxscale(self.X_train, self.samples, self.X_test)
        today_date = datetime.today().strftime('%Y%m%d')
        df = pd.DataFrame(reversed_x_val, columns=self.columns_x_val, index=self.samples_index)
        #add similarty scores:
        self.sample_score.index = df.index
        df['sample_scoring'] = self.sample_score
        data = {'date': today_date,
                'model_type': self.model_type}
        if type(self.model_object) is list:
            if len(self.model_object) == 2:
                data['algorithm_1'] = str(self.model_object[0].model_type)
                data['algorithm_2'] = str(self.model_object[1].model_type)
            elif len(self.model_object) == 3:
                data['algorithm_1'] = str(self.model_object[0].model_type)
                data['algorithm_2'] = str(self.model_object[1].model_type)
                data['algorithm_3'] = str(self.model_object[2].model_type)

        data['density_method'] = self.density_method
        data['method_threshold'] = self.method_threshold

        df_info = pd.DataFrame.from_dict(data, orient='index').T
        col_move = len(df_info.columns)
        df_scores = pd.DataFrame.from_dict(self.scores, orient='index').T
        df_scores['average_score'] = df_scores[1:].sum(axis=1) / len(self.model_object)

        df_classification = pd.DataFrame()
        # temp = list(df.select_dtypes('float').columns.values)
        # df.apply(lambda x: round(x, 2))

        writer = pd.ExcelWriter(
            str(self.models_algorithms) + '_Ouput_Selection_' + str(self.select.__name__) + '_' + str(today_date) + '.xlsx',
            engine='xlsxwriter')
        df_info.to_excel(writer,
                         index=False, startrow=1)
        df_scores.to_excel(writer,
                           index=False, startcol=col_move)

        if self.classification == 1:
            labels = ["True Neg", "False Pos", "False Neg", "True Pos"]
            categ = ["Zero", "One"]
            df_precision_score = pd.DataFrame.from_dict(self.precision_scores, orient='Index')
            df_precision_score.to_excel(writer,
                                        index=True,
                                        startrow=3)

        df.to_excel(
            writer,
            index=True, startrow=13)
        if self.classification == 1:
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
        writer.save()

        # selection_probas_val = pd.DataFrame(self.selection_probas_val)
        # selection_probas_val.to_csv('test')


# models = [SvmModel, RfModel, CBModel]
models = [SVR_Model, RandomForestEnsemble
    , CatBoostReg
          ]
# selection_functions = [selection_functions]

selection = SelectionFunction()
alg = Algorithm(models, selection, select=max_std_sampling, model_type='Regression')
alg.run_algorithm()
alg.compare_query_changes()
alg.similairty_scoring(method='gower', threshold=0.25, n_instances=100)
alg.output_data()

# alg.run('Results_Complete.csv', 'uncertainty_sampling', 1889244, True, model_type='Classification',
# limit=-1)
