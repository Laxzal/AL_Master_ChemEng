##### ----Modules--------
import os
import pandas as pd
import pprint
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib notebook
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC

import joblib

# from sklearn.externals import joblib


print(os.name)

###Change Directory
wrk_path_1 = r"C:\Users\calvi\OneDrive\Documents\2020\Liposomes Vitamins\LiposomeFormulation"
wrk_path_2 = r"C:\Users\Calvin\OneDrive\Documents\2020\Liposomes Vitamins\LiposomeFormulation"
wrk_path_3 = r"/Users/calvin/Documents/OneDrive/Documents/2020/Liposomes Vitamins/LiposomeFormulation"

if os.name == 'posix':
    os.chdir(wrk_path_3)
    print("Utilising MacBook")
else:
    try:
        os.chdir(wrk_path_2)
        print("Utilising Home Pathway")
    except OSError:
        os.chdir(wrk_path_1)
        print("Utilising Lab Pathway")

## Import Files

datafile = pd.read_csv('Results_Complete.csv')
datafile.corr()
print(list(datafile.columns))
print("Number of Columns:", len(datafile.columns.unique()))



# Drop Columns - Not Useful

    # List of columns to drop
drop_list = ['Duplicate_Check',
             'PdI Width (d.nm)',
             'PdI',
             'Z-Average (d.nm)']


datafile_red = datafile.drop(drop_list, axis = 1).reset_index(drop = True)
print((datafile_red.head()))


# Target Column Information & Encode
    #First, let us see what values exist in the target column and return a count of NaN values
print(datafile_red['ES_Aggregation'].unique())
print(datafile_red['ES_Aggregation'].isnull().sum(axis=0))
datafile_cleaned = datafile_red[datafile_red['ES_Aggregation'].notna()].reset_index(drop=True)

print(datafile_cleaned)
print(datafile_cleaned['ES_Aggregation'].unique())

ax = sns.countplot(x='ES_Aggregation', data=datafile_red)
print(datafile_red['ES_Aggregation'].value_counts())

