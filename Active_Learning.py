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
import os
import pandas as pd
import numpy as np
from datetime import datetime
from CommitteeClass import Committee
from DataLoad import data_load
from Models import SvmModel, RfModel
from QueriesCommittee import KLMaxDisagreement, max_disagreement_sampling, vote_entropy, vote_entropy_sampling
from QueryInstanceDensityWeighted import QueryInstanceDensityWeighted
from SelectionFunctions import SelectionFunction
from SplitData import split_data
from Standardisation import MinMaxScaling
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

    def run(self, file, select: str = 'entropy_sampling', instances: int = 1, randomize_tie_break: bool = False,
            hide: str = None):
        '''

        :param file: CSV format of the experiment results
        :param select: Selection of the sampling method used on the proba results. Entropy/Margin/Uncertainty
        :param instances: Number of return suggested experiments to be conducted
        :param randomize_tie_break: If the selection of the returned experiment suggestions are at equal position due
                to the sampling, then the randomized tie-break will randomly choose
        :param hide: Choose a component to hide during the training of the algorithm.
        :return: A CSV file of experiments that the algorithm needs to be learn better
        '''
        # Pull in unlabelled Data

        X_val, columns_x_val = unlabelled_data('stratified_sample_experiment.csv',
                                               method='fillna')  # TODO need to rename

        # Run Split

        datagather = data_load()
        datafile = datagather.read_file(file)
        # datagather.datafile_info(datafile)
        cleaning = datagather.drop_columns()
        cleaned = datagather.target_check()

        object2 = split_data()
        object2.labelencode()
        object2.temp()
        object2.filter_table(hide)
        object2.x_array()
        object2.y_array()

        X_train, X_test, y_train, y_test = object2.split_train_test(0.2)

        # normalise data - Increased accuracy from 50% to 89%
        normaliser = MinMaxScaling()
        X_train, X_val, X_test = normaliser.normalise(X_train, X_val, X_test)

        if type(self.model_object) is list:
            self.committee_models = Committee(self.model_object)
            self.committee_models.gridsearch_commitee(X_train, y_train, 'balanced', splits=5)
            self.committee_models.fit_data(X_train, y_train)
            probas_val = self.committee_models.vote_proba(X_val)
            self.committee_models.vote(X_val)
            selection_probas_val = self.committee_models.query(self.committee_models, vote_entropy_sampling,
                                                               X_val, n_instances=instances)
            model_type = self.committee_models.printname()
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
        # TODO Fix up this code completely - Choosing the right key needs to be automatic, not a guess
        _, reversed_x_val, _ = normaliser.inverse_normalise(X_test, selection_probas_val[1], X_test)
        today_date = datetime.today().strftime('%Y%m%d')
        df = pd.DataFrame(reversed_x_val, columns=columns_x_val, index=selection_probas_val[0])
        df.to_csv(
            str(model_type) + '_Ouput_Selection_' + str(select) + '_' + str(today_date) + '.csv',
            index=True)

        # test = DensityWeightedUncertaintySampling(X_uld=X_val, X_lld=X_test, y_labelled=y_test)
        # test.prob_x()
        # ids = test.make_query()

        test = QueryInstanceDensityWeighted(X_val, X_test, y_test, 'entropy', distance='euclidean', beta = 1.0)
        test.select(batch_size=1, proba_prediction=probas_val)

models = [SvmModel, RfModel]
# , CBModel]
# selection_functions = [selection_functions]

selection = SelectionFunction()
alg = Algorithm(models[0], selection)
alg.run('Results_Complete.csv', 'uncertainty_sampling', 10, True)
