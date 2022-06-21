import os
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from Data_Analysis import Data_Analyse


class Data_Load_Old(object):
    column_drop = ['Duplicate_Check',
                   'PdI Width (d.nm)',
                   'PdI',
                   'Z-Average (d.nm)']
    datafile = None
    datafile_cleaned = None

    def __init__(self):
        self.datafile = None
        self.datafile_clean = None

    def read_file(self, file) -> pd.DataFrame:
        datafile = pd.read_csv(str(file))
        print(datafile.head())
        self.datafile = datafile
        return self.datafile

    def datafile_info(self, datafile):
        print(list(datafile.columns))
        print("Number of Column: ", len(datafile.columns.unique()))

    def drop_columns(self):
        datafile_clean = self.datafile.drop(self.column_drop, axis=1).reset_index(drop=True)
        print(list(datafile_clean.columns))
        print("Number of Column: ", len(datafile_clean.columns.unique()))
        self.datafile_clean = datafile_clean
        return self.datafile_clean

    def target_check(self):
        # First let's see what values exist in the target column ('ES_Aggregation') and return a count of NaN values
        print(self.datafile_clean['ES_Aggregation'].unique())
        print(self.datafile_clean['ES_Aggregation'].isnull().sum(axis=0))
        datafile_cleaned = self.datafile_clean[self.datafile_clean['ES_Aggregation'].notna()].reset_index(drop=True)
        # print(datafile_cleaned)
        print(datafile_cleaned['ES_Aggregation'].unique())
        # ax = sns.countplot(x='ES_Aggregation', data=datafile_cleaned)
        # plt.show()
        print(datafile_cleaned['ES_Aggregation'].value_counts())
        # print(datafile_cleaned)
        Data_Load_Old.datafile_cleaned = datafile_cleaned
        return Data_Load_Old.datafile_cleaned


class Data_Load_Split(object):

    def __init__(self, file, hide_component: str = None, alg_categ: str = None,
                 split_ratio: float = 0.2,
                 shuffle_data: bool = True, drop_useless_columns: bool = True,
                 filter_target: bool=False, target: str='Z-Average (d.nm)', smaller_than: float=1000.0,
                 column_removal_experiment: list=None
                 ):

        assert alg_categ in ['Regression', 'Classification', 'Regression and Classification', 'Reg&Class',
                             'MultiOutput Regression', 'MO Regression']

        self.file = file
        self.hide_component = hide_component
        self.alg_categ = alg_categ
        self.split_ratio = split_ratio
        self.shuffle_data = shuffle_data
        self.drop_useless_columns = drop_useless_columns
        self.regression_table_drop = ['ES_Aggregation',
                                      'PdI Width (d.nm)',
                                      'PdI',
                                      'Duplicate_Check']

        self.classification_table_drop = ['PdI Width (d.nm)',
                                          'PdI',
                                          'Z-Average (d.nm)',
                                          'Duplicate_Check']
        self.multi_regression_table_drop = ['ES_Aggregation',
                                            'Duplicate_Check']



        self.datafile = None
        self.train_table = None
        self.dum = None
        self.X = None
        self.y = None
        self.hide = None
        self.columns_converted = []
        self.filter_target = filter_target
        self.target = target
        self.smaller_than = smaller_than
        self.dummation_occured = 0
        self.column_removal_experiment = column_removal_experiment
        ###Functions to be run Automatically
        self.initial_read_file()
        self.label_encode()
        self.dummation_groupby()
        self.filter_table()
        self.alg_category()
        self.initial_x_array()
        self.inital_y_array()
        self.class_names_str = None

    def initial_read_file(self):
        try:
            self.datafile = pd.read_csv(str(self.file))
        except Exception:
            self.datafile = pd.read_excel(str(self.file))


        #if self.drop_useless_columns == True:
            #self.datafile = self._useless_column_drop(self.datafile)

        if self.filter_target == True:
            self.datafile = self._target_filter(dataframe=self.datafile,target=self.target, smaller_than=self.smaller_than)

        return self.datafile

    def _target_filter(self, dataframe: pd.DataFrame, target: str, smaller_than: float):
        self.datafile = dataframe[dataframe[str(target)] <= float(smaller_than)]

        return self.datafile


    #def _useless_column_drop(self, dataframe: pd.DataFrame):
        #self.datafile = dataframe.drop(columns=self.drop_columns_useless)

        #return self.datafile

    def label_encode(self):
        if self.alg_categ in {'Classification'}:
            lb = LabelEncoder()
            self.datafile = self.datafile[self.datafile['ES_Aggregation'].notna()].reset_index(drop=True)
            self.datafile['ES_Aggregation_encoded'] = lb.fit_transform((self.datafile['ES_Aggregation']))
            print(self.datafile['ES_Aggregation_encoded'].value_counts())
            self.class_names_str = lb.classes_
        elif self.alg_categ in {'Regression'}:
            self.datafile = self.datafile[self.datafile['Z-Average (d.nm)'].notna()].reset_index(drop=True)

        elif self.alg_categ in {'MultiOutput Regression', 'MO Regression'}:
            self.datafile = self.datafile[self.datafile['PdI Width (d.nm)',
                                                        'PdI',
                                                        'Z-Average (d.nm)'].notna()].reset_index(drop=True)

    def dummation_groupby(self) -> pd.DataFrame:
        if "Component_1" and "Component_2" and "Component_3" in self.datafile.columns:
            self.dum = pd.get_dummies(self.datafile, columns=['Component_1', 'Component_2', 'Component_3'],
                                      prefix="", prefix_sep="")
            # TODO Add in Component 4 into 'columns = ' when it becomes relevant. Currently not relevant due to PEG
            self.dum = self.dum.groupby(level=0, axis=1, sort=False).sum()
            self.dummation_occured = 1
        else:
            self.dum = self.datafile.copy()
    def filter_table(self):
        if self.hide_component is not None:
            self.hide = self.dum[self.dum[str(self.hide_component)] == 1]
            self.train_table = self.dum[self.dum[str(self.hide_component)] == 0]
            return self.train_table, self.hide
        else:
            pass

    def alg_category(self):

        # Need to think of a better way to deal

        if self.alg_categ in {'Regression'}:
            if self.dum is not None and self.train_table is not None:
                self.train_table.drop(self.regression_table_drop, axis=1, inplace=True)
                self.dum.drop(self.regression_table_drop, axis=1, inplace=True)
            else:
                self.dum.drop(self.regression_table_drop, axis=1, inplace=True)
                self.datafile.drop(self.regression_table_drop, axis=1, inplace=True)


        elif self.alg_categ in {'Classification'}:
            if self.dum is not None and self.train_table is not None:
                self.train_table.drop(self.classification_table_drop, axis=1, inplace=True)
                self.dum.drop(self.classification_table_drop, axis=1, inplace=True)
            else:
                self.dum.drop(self.classification_table_drop, axis=1, inplace=True)
                self.datafile.drop(self.classification_table_drop, axis=1, inplace=True)

        elif self.alg_categ in {'Regression and Classification', 'Reg&Class'}:
            print('Needs to be implemented...')

        elif self.alg_categ in {'MultiOutput Regression', 'MO Regression'}:
            if self.dum is not None and self.train_table is not None:
                self.train_table.drop(self.multi_regression_table_drop, axis=1, inplace=True)
                self.dum.drop(self.multi_regression_table_drop, axis=1, inplace=True)
            else:
                self.dum.drop(self.multi_regression_table_drop, axis=1, inplace=True)
                self.datafile.drop(self.multi_regression_table_drop, axis=1, inplace=True)
        else:
            print('What did you write that got you past the assertion check...')

    def _column_removal_(self, column_removal_experiment: list=None):
        if self.dum is not None and self.train_table is not None:
            self.train_table.drop(columns=column_removal_experiment)
            self.dum.drop(columns=column_removal_experiment)
        elif self.train_table is None:
            self.dum.drop(columns=column_removal_experiment, inplace=True)



    def initial_x_array(self):

        if self.column_removal_experiment is not None:
            self._column_removal_(self.column_removal_experiment)

        if self.dum is not None and self.train_table is not None:
            if self.alg_categ in {'Classification'}:
                x_table = self.train_table.drop(['ES_Aggregation_encoded', 'ES_Aggregation'], axis=1).reset_index(
                    drop=True)
            elif self.alg_categ in {'Regression'}:
                x_table = self.train_table.drop(['Z-Average (d.nm)'], axis=1).reset_index(drop=True)

            elif self.alg_categ in {'MultiOutput Regression', 'MO Regression'}:
                x_table = self.train_table.drop(['PdI Width (d.nm)', 'PdI', 'Z-Average (d.nm)'],
                                                axis=1).reset_index(drop=True)

            self.X = x_table.values


        elif self.train_table is None:
            if self.alg_categ in {'Classification'}:
                x_table = self.dum.drop(['ES_Aggregation_encoded', 'ES_Aggregation'], axis=1).reset_index(drop=True)
            elif self.alg_categ in {'Regression'}:
                x_table = self.dum.drop(['Z-Average (d.nm)'], axis=1).reset_index(drop=True)

            elif self.alg_categ in {'MultiOutput Regression', 'MO Regression'}:
                x_table = self.dum.drop(['PdI Width (d.nm)', 'PdI', 'Z-Average (d.nm)'],
                                        axis=1).reset_index(drop=True)
            self.X = x_table.values
        #TODO Need to fix this at some point when I do dummy grouping again
        if self.dummation_occured == 1:
            for i in x_table.columns:
                if (x_table[str(i)].isin([0, 1]).all()) == True:
                    self.columns_converted.append(True)
                else:
                    self.columns_converted.append(False)
        else:
            for i in range(len(x_table.columns)):
                self.columns_converted.append(False)


        return self.X



    def inital_y_array(self):
        if self.dum is not None and self.train_table is not None:

            if self.alg_categ in {'Classification'}:
                self.y = self.train_table['ES_Aggregation_encoded'].values
            elif self.alg_categ in {'Regression'}:
                self.y = self.train_table['Z-Average (d.nm)'].values
            elif self.alg_categ in {'MultiOutput Regression', 'MO Regression'}:
                self.y = self.train_table['PdI Width (d.nm)', 'PdI', 'Z-Average (d.nm)'].values


        elif self.train_table is None:
            if self.alg_categ in {'Classification'}:
                self.y = self.dum['ES_Aggregation_encoded'].values
            elif self.alg_categ in {'Regression'}:
                self.y = self.dum['Z-Average (d.nm)'].values
            elif self.alg_categ in {'MultiOutput Regression', 'MO Regression'}:
                self.y = self.dum['PdI Width (d.nm)', 'PdI', 'Z-Average (d.nm)'].values
        return self.y

    def analyse_data(self, save_path, column_names, plot):
        data_analyse = Data_Analyse()

        data_analyse.histogram(self.y, data_name='y_all', save_path=save_path, plot=plot)

        data_analyse.qqplot_data(self.y, data_name='y_all', save_path=save_path, plot=plot)

        print('----Shapiro Wilk Y Train----')
        self.shapiro_wilk_y_train = data_analyse.shapiro_wilk_test(self.y)

        print(self.shapiro_wilk_y_train)
        print('-----------------------------')

        print('----Dagostino K^2 Y Train----')
        self.dagostino_k2_y, self.dagostino_p_y, self.dagiston_is_gaussian = data_analyse.dagostino_k2(self.y)

        print('----Anderson Y Train----')
        self.anderson_darling_train = data_analyse.anderson_darling(self.y)

        print('----Heatmap X Train----')
        self.heatmap_train = data_analyse.heatmap(self.X, column_names, data_name='x_all', save_path=save_path,
                                                  plot=False)

        print('----Box Plot X Train----')
        self.box_plot_train = data_analyse.box_plot(self.X, column_names, data_name='x_all', save_path=save_path,
                                                    plot=False)

        print('----Variance Inflation Factor_X_train----')
        self.variance_inflation_factor_x_train = data_analyse.variance_inflation_factor(self.X, column_names)
        print(self.variance_inflation_factor_x_train)
        file_name = os.path.join(save_path,"variance_inflation_factor")
        self.variance_inflation_factor_x_train.to_csv(str(file_name) + ".csv", index=False)


        print("------Sweet_Viz---------")
        #self.sweet_viz = data_analyse.sweet_viz(self.X,column_names,save_path=save_path)

        print("------Target Included---------")
        temp_df = pd.DataFrame(self.X,columns=column_names)
        temp_df['Average_Size'] = self.y

        self.sweet_viz_target = data_analyse.sweet_viz(temp_df,feature_names=None,target="Average_Size", save_path=save_path)

        print("------Pandas_Profile---------")
        temp_df=pd.DataFrame(self.X,columns=column_names)
        data_analyse.pandas_profiling(temp_df,save_path=save_path)


        return self.dagiston_is_gaussian

    def split_train_test(self):

        # TODO Need to look into the stratify parameter - if function again...
        if self.alg_categ in {'Classification'}:
            (X_train, X_test, y_train, y_test) = \
                train_test_split(self.X, self.y, test_size=self.split_ratio, random_state=42, shuffle=self.shuffle_data,
                                 stratify=self.y)
        else:
            (X_train, X_test, y_train, y_test) = \
                train_test_split(self.X, self.y, test_size=self.split_ratio, random_state=42, shuffle=self.shuffle_data)

        if self.hide is not None:
            if self.alg_categ in {'Classification'}:
                x_temp = self.hide.drop(['ES_Aggregation_encoded', 'ES_Aggregation'], axis=1).reset_index(drop=True)
                x_temp = x_temp.values
                X_test = np.vstack([X_test, x_temp])
                y_temp = self.hide['ES_Aggregation_encoded'].values
                y_test = np.hstack([y_test, y_temp])
            elif self.alg_categ in {'Regression'}:
                x_temp = self.hide.drop(['Z-Average (d.nm)'], axis=1).reset_index(drop=True)
                x_temp = x_temp.values
                X_test = np.vstack([X_test, x_temp])
                y_temp = self.hide['Z-Average (d.nm)'].values
                y_test = np.hstack([y_test, y_temp])
            elif self.alg_categ in {'MultiOutput Regression', 'MO Regression'}:
                x_temp = self.hide.drop(['PdI Width (d.nm)', 'PdI', 'Z-Average (d.nm)'], axis=1).reset_index(drop=True)
                x_temp = x_temp.values
                X_test = np.vstack([X_test, x_temp])
                y_temp = self.hide['PdI Width (d.nm)', 'PdI', 'Z-Average (d.nm)'].values
                y_test = np.hstack([y_test, y_temp])

        return X_train, X_test, y_train, y_test
