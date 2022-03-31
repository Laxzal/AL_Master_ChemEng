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
from random import shuffle
from typing import Callable

import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.metrics import *

from BaseComittee import BaseLearner
from BatchMode import uncertainty_batch_sampling
from CommitteeClass import CommitteeRegressor, CommitteeClassification
from DataLoad import Data_Load_Split
from DiversitySampling_Clustering import DiversitySampling
from Models import SvmModel, RfModel, RandomForestEnsemble, SVR_Model, CBModel
from QueriesCommittee import KLMaxDisagreement, max_disagreement_sampling, vote_entropy, vote_entropy_sampling, \
    max_std_sampling
from QueryInstanceDensityWeighted import QueryInstanceDensityWeighted
from SelectionFunctions import SelectionFunction

from PreProcess import MinMaxScaling
from TrainModel import TrainModel
import scipy.sparse as sp
from DensityWeightedUncertaintySampling import DensityWeightedUncertaintySampling, DensityWeightedLogisticRegression

"""'

GATHER DATA

'"""
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
y = None

'''

Selection

'''


class Algorithm(object):
    accuracies = []

    def __init__(self, model_object, selection_functions):
        self.model_test = None
        self.optimised_classifier = None
        self.model_object = model_object
        self.selection_functions = selection_functions

        self.diversity_sampling = DiversitySampling()

    def run(self, file, select: Callable = SelectionFunction.entropy_sampling, instances: int = 1889244,
            randomize_tie_break: bool = True,
            hide: str = None, model_type: str = 'Classification', limit: int = -1, max_epochs: int = 10,
            num_clusters: int = 5, perc_uncertain: float = 0.1):
        '''

        :param file: CSV format of the experiment results
        :param select: Selection of the sampling method used on the proba results. Entropy/Margin/Uncertainty
        :param instances: Number of return suggested experiments to be conducted
        :param randomize_tie_break: If the selection of the returned experiment suggestions are at equal position due
                to the sampling, then the randomized tie-break will randomly choose
        :param hide: Choose a component to hide during the training of the algorithm.
        :return: A CSV file of experiments that the algorithm needs to be learn better
        '''
        assert perc_uncertain <= 1
        assert model_type in ['Classification', 'Regression']
        # Pull in unlabelled Data

        X_val, columns_x_val = unlabelled_data('unlabelled_data_full.csv',
                                               method='fillna')  # TODO need to rename


        if limit > 0:
            shuffle(X_val)
            X_val = X_val[:limit]

        uncertain_count = math.ceil(len(X_val) * perc_uncertain)

        # Run Split

        data_load_split = Data_Load_Split(file, hide_component=None, alg_categ=model_type, split_ratio=0.2,
                                          shuffle_data=True)

        X_train, X_test, y_train, y_test = data_load_split.split_train_test()

        # normalise data - Increased accuracy from 50% to 89%
        normaliser = MinMaxScaling()
        normaliser.minmaxscale_fit(X_train, X_test, X_val)
        X_train, X_val, X_test = normaliser.minmaxscale_trans(X_train, X_val, X_test)

        if model_type == 'Regression':
            if type(self.model_object) is list:
                self.committee_models = CommitteeRegressor(self.model_object, X_train, X_test, y_train, y_test, X_val,
                                                           splits=3,
                                                           kfold_shuffle=True,
                                                           scoring_type='r2',
                                                           instances=instances,
                                                           query_strategy=max_std_sampling)
                self.committee_models.gridsearch_committee()
                self.committee_models.fit_data()
                self.committee_models.score()
                selection_probas_val = self.committee_models.query(self.committee_models)


        elif model_type == 'Classification':
            if type(self.model_object) is list:
                self.committee_models = CommitteeClassification(self.model_object, X_train, X_test, y_train, y_test,
                                                                X_val,
                                                                vote_entropy_sampling,
                                                                c_weight='balanced',
                                                                splits=3,
                                                                scoring_type='precision',
                                                                kfold_shuffle=True)
                self.committee_models.gridsearch_committee()
                self.committee_models.fit_data()
                probas_val = self.committee_models.vote_proba()
                self.committee_models.vote()
                selection_probas_val = self.committee_models.query(self.committee_models, n_instances=instances,
                                                                   X_labelled=X_train)
                model_type = self.committee_models.printname()


                centroids, outliers, randoms = self.diversity_sampling.get_cluster_samples(selection_probas_val, num_clusters=num_clusters, limit = -1)
                self.diversity_sampling.graph_clusters(selection_probas_val)

                samples = centroids + outliers + randoms

            else:

                ### GridSearch/Fit Data - Single Algorithm ###
                self.model_test = TrainModel(self.model_object)
                self.optimised_classifier = self.model_test.optimise(X_train, y_train, 'balanced', splits=5,
                                                                     scoring='precision')
                (X_train, X_val, X_test) = self.model_test.train(X_train, y_train, X_val, X_test)
                probas_val = \
                    self.model_test.predictions(X_train, X_val, X_test)
                self.model_test.return_accuracy(1, y_test, y_train)
                model_type = self.model_object.model_type
                ##Attempt to get PROBA values of X_Val

                selection_probas_val = \
                    self.selection_functions.select(self.optimised_classifier, X_val, instances, randomize_tie_break)

                if select == 'margin_sampling':
                    selection_probas_val = \
                        self.selection_functions.margin_sampling(self.optimised_classifier, X_val, instances,
                                                                 randomize_tie_break)
                elif select == 'entropy_sampling':
                    selection_probas_val = \
                        self.selection_functions.entropy_sampling(self.optimised_classifier, X_val, instances,
                                                                  randomize_tie_break)
                elif select == 'uncertainty_sampling':
                    selection_probas_val = \
                        self.selection_functions.uncertainty_sampling(self.optimised_classifier, X_val, instances,
                                                                      randomize_tie_break)

        # print('val predicted:',
        #      self.model_test.val_y_predicted.shape,
        #      self.model_test.val_y_predicted)
        print(probas_val)

        print('probabilities:', probas_val.shape, '\n',
              np.argmax(probas_val, axis=1))
        print('the unique values in the probability values is: ', np.unique(probas_val))
        print('size of unique values in the probas value array: ', np.unique(probas_val).size)

        print('SHAPE OF X_TRAIN', X_train.shape[0])
        # print(self.optimised_classifier.classes_)
        # print(selection_probas_val)
        similarity_scores = []
        index = []
        data_info = []
        for _, data in enumerate(samples):
            for _, data_x in enumerate(data):
                similarity_scores.append(data_x[0])
                index.append(data_x[1])
                data_info.append(data_x[2])




        # TODO Fix up this code completely - Choosing the right key needs to be automatic, not a guess
        _, reversed_x_val, _ = normaliser.inverse_minmaxscale(X_train, data_info, X_test)
        today_date = datetime.today().strftime('%Y%m%d')
        df = pd.DataFrame(reversed_x_val, columns=columns_x_val, index=similarity_scores)
        #temp = list(df.select_dtypes('float').columns.values)
        #df.apply(lambda x: round(x, 2))
        df.to_csv(
            str(model_type) + '_Ouput_Selection_' + str(select) + '_' + str(today_date) + '.csv',
            index=True)

        # test = DensityWeightedUncertaintySampling(X_uld=X_val, X_lld=X_test, y_labelled=y_test)
        # test.prob_x()
        # ids = test.make_query()

        # test = QueryInstanceDensityWeighted(X_val, X_test, y_test, 'entropy', distance='euclidean', beta = 1.0)
        # test.select(batch_size=1, proba_prediction=probas_val)

        # x, y = uncertainty_batch_sampling(self.optimised_classifier,X= X_val,X_labelled=X_train,n_jobs=-1)

        # test = BaseLearner(self.optimised_classifier, vote_entropy_sampling, X_train, y_train)
        # test._fit_new()





models = [SvmModel, RfModel]
          #,CBModel]
#models = [SVR_Model, RandomForestEnsemble]
# , CBModel]
# selection_functions = [selection_functions]

selection = SelectionFunction()
alg = Algorithm(models, selection)
alg.run('Results_Complete.csv', 'uncertainty_sampling', 1889244, True, model_type='Classification',
        limit = -1)
