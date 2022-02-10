import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import DataLoad


class split_data(DataLoad.data_load):
    X = None
    y = None

    def __init__(self):
        self.x_table = None
        self.datafile_pre_split = None
        self.dum = None
        self.train_table = None
        self.hide = None
        split_data.datafile = DataLoad.data_load.datafile_cleaned

    def labelencode(self):
        lb = LabelEncoder()
        split_data.datafile['ES_Aggregation_encoded'] = lb.fit_transform(
            (split_data.datafile_cleaned['ES_Aggregation']))
        print(split_data.datafile_cleaned['ES_Aggregation_encoded'].value_counts())

        return split_data.datafile_cleaned

    def temp(self) -> pd.DataFrame:
        if 'ES_Aggregation_encoded' in split_data.datafile_cleaned.columns:
            self.dum = pd.get_dummies(self.datafile_cleaned, columns=["Component_1", "Component_2", "Component_3"],
                                      prefix="", prefix_sep="")
            self.dum = self.dum.groupby(level=0, axis=1, sort=False).sum()

            return self.dum
        else:
            print("Please ensure label encoding has been completed (split_data.labelencode()")

    def filter_table(self, user_input: str = None) -> pd.DataFrame:
        if user_input is not None:
            self.hide = self.dum[self.dum[str(user_input)] == 1]
            self.train_table = self.dum[self.dum[str(user_input)] == 0]
            return self.train_table, self.hide
        else:
            pass

    def x_array(self) -> np.ndarray:
        if self.train_table is not None:
            x_table = self.train_table.drop(['ES_Aggregation_encoded', 'ES_Aggregation'], axis=1).reset_index(drop=True)
            split_data.X = x_table.values
            return split_data.X
        elif self.train_table is None:
            self.x_table = self.dum.drop(['ES_Aggregation_encoded', 'ES_Aggregation'], axis=1).reset_index(drop=True)
            split_data.X = self.x_table.values
            return split_data.X
        else:
            print("Error with x_array function")

    def y_array(self) -> np.ndarray:
        if self.train_table is not None:
            split_data.y = self.train_table['ES_Aggregation_encoded'].values
            return split_data.y
        elif 'ES_Aggregation_encoded' in self.dum.columns:
            split_data.y = self.dum['ES_Aggregation_encoded'].values
            # print(y)
            return split_data.y
        else:
            print('Please label encode')

    def split_train_test(self, split_ratio, shuffle_data=True):
        # TODO Implement stratification option
        '''
        :param split_ratio:
        :param shuffle_data:
        :param hide:  This parameter can be any component from the component list within the DataFrame. Example: HSPC.
        So if user writes 'HSPC' for the parameter hide, then the code will extract component from the DataFrame prior
        to splitting. After splitting, the hidden component will be appended to the Test Data
        :return:
        '''
        # HERE
        # if hide is not None:
        # extract whaytever hide says
        # else:
        # pass
        (X_train, X_test, y_train, y_test) = \
            train_test_split(split_data.X, split_data.y, test_size=split_ratio, random_state=42, shuffle=shuffle_data,
                             stratify=split_data.y)

        if self.hide is not None:
            x_temp = self.hide.drop(['ES_Aggregation_encoded', 'ES_Aggregation'], axis=1).reset_index(drop=True)
            x_temp = x_temp.values
            X_test = np.vstack([X_test, x_temp])
            y_temp = self.hide['ES_Aggregation_encoded'].values
            y_test = np.hstack([y_test, y_temp])
        # TODO Fix this print code for np array
        # print("Y_train dataset:")
        # print(y_train.value_counts())
        # print('Classification: No, Population (%): ', 100 * (y_train.value_counts()[0]) / (y.value_counts()[0]))
        # print('-----------')

        # print("Y_test dataset:")
        # print(y_test.value_counts())
        # print('Classification: No, Population (%): ', 100 * (y_test.value_counts()[0]) / (y.value_counts()[0]))
        return X_train, X_test, y_train, y_test