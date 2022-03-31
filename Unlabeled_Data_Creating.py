import itertools
import os
import pandas as pd

wrk_path = r"/Users/calvin/Documents/MastersProject"
os.chdir(wrk_path)
'''
Creating Component 1 - DataFrame (Phospholipids)
1. HSPC
2. POPC
3. DOPC
'''

component_1_df = pd.DataFrame(columns=['component_1'])
list_component_lipids = ['hspc', 'popc', 'dopc']
component_1_df['component_1'] = list_component_lipids

'''
Creating Component 2 DataFrame (Drugs/Vitamins)
The list of components are:

Vitamin E
Vitamin D
Cholesterol
This list is to be expanded at a later date. Furthermore, there is discussion around attempting to encapsulate two of these components at a time
'''

component_2_df = pd.DataFrame(columns=['component_2'])
# list_component_drug = ['vite','vitd','chol', 'vita','vitb12']
list_component_drug = ['vite', 'vitd', 'chol']
component_2_df['component_2'] = list_component_drug

'''
Creating Component 3 DataFrame (Stealth Polymer)
The list of components are:

PEG - 2000
This list may be expanded at a later date
'''

component_3_df = pd.DataFrame(columns=['component_3'])
list_component_polymer = ['peg2000']
component_3_df['component_3'] = list_component_polymer

'''
Creating Vol by Vol Percent For Component 1 DataFrame
In general, this percent of component 1 must always be equal to or over 50% of the formulation. However, this is not a scientific rule.
'''

vol_vol_pcnt_1_df = pd.DataFrame(columns=['vol_vol_pcnt_1'])
vol_vol_pcnt = [90, 80, 70, 60, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5]
vol_vol_pcnt_1_df['vol_vol_pcnt_1'] = [i for i in vol_vol_pcnt if i >= 50]

'''
Creating Vol by Vol Percent For Component 2 DataFrame
This will have to be below or equal to 50%
'''

vol_vol_pcnt_2_df = pd.DataFrame(columns=['vol_vol_pcnt_2'])
vol_vol_pcnt_2_df['vol_vol_pcnt_2'] = [i for i in vol_vol_pcnt if i <= 50]

'''
Creating Vol by Vol Percent For Component 3 DataFrame
This will have to be below or equal to 5%
'''
vol_vol_pcnt_3_df = pd.DataFrame(columns=['vol_vol_pcnt_3'])
vol_vol_pcnt_3_df['vol_vol_pcnt_3'] = [i for i in vol_vol_pcnt if i <= 5]

'''
Creating the Concentration DataFrames for each component
The chosen concentrations (mM) are:

75
50
30
25
20
10
5
'''

conc = [75, 50, 30, 25, 20, 10, 5]
conc_1_df = pd.DataFrame(columns=['conc_1'])
conc_1_df['conc_1'] = conc
conc_2_df = pd.DataFrame(columns=['conc_2'])
conc_2_df['conc_2'] = conc
conc_3_df = pd.DataFrame(columns=['conc_3'])
conc_3_df['conc_3'] = conc

'''
Creating the dispense speed dataframe
While the speed can be anywhere between 1-400ul/s, only 3 speeds shall be chosen to reduce the number of experiments required
'''
# speed = [400,300,200,120,80,30]
speed = [400, 120, 30]
speed_disp_df = pd.DataFrame(columns=['speed_disp'])
speed_disp_df['speed_disp'] = speed

'''
Creating the Formulation Combination DataFrame
'''

formulation_combination_df = pd.DataFrame(list(itertools.product(component_1_df.component_1,
                                                                 component_2_df.component_2,
                                                                 component_3_df.component_3,
                                                                 vol_vol_pcnt_1_df.vol_vol_pcnt_1,
                                                                 vol_vol_pcnt_2_df.vol_vol_pcnt_2,
                                                                 vol_vol_pcnt_3_df.vol_vol_pcnt_3,
                                                                 conc_1_df.conc_1,
                                                                 conc_2_df.conc_2,
                                                                 conc_3_df.conc_3,
                                                                 speed_disp_df.speed_disp
                                                                 )),
                                          columns=['component_1',
                                                   'component_2',
                                                   'component_3',
                                                   'vol_vol_pcnt_1',
                                                   'vol_vol_pcnt_2',
                                                   'vol_vol_pcnt_3',
                                                   'conc_1',
                                                   'conc_2',
                                                   'conc_3',
                                                   'speed_disp'])

'''
Total vol by vol percent column
The three vol by vol percent columns are to be summed. Then totals that are higher or lower than 100 must be removed (as
 it is a percent and you cannot go over 100, nor should the percent be lower than 100)
'''

col_list = list(formulation_combination_df)
rmve = ('component_1',
        'component_2',
        'component_3',
        'conc_1',
        'conc_2',
        'conc_3',
        'speed_disp')
for i in range(len(rmve)):
    col_list.remove(rmve[i])

formulation_combination_df['total_pcnt'] = formulation_combination_df[col_list].sum(axis=1)
formulation_combination_df = formulation_combination_df.loc[formulation_combination_df['total_pcnt'] == 100].reset_index(drop=True)

print (formulation_combination_df.info())
print("DONE")

formulation_combination_df.to_csv('full_unlabeled_combination_experiment.csv', index = False)