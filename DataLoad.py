import pandas as pd


class data_load(object):
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
        data_load.datafile_cleaned = datafile_cleaned
        return data_load.datafile_cleaned
