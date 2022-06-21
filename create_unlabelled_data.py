''"""

Create a set of unlabelled data

"""''

"""Import Modules"""

import os
import pandas as pd
import numpy as np
import itertools

"""
Import dataframe
"""
wrk_path_3 = r"/Users/calvin/Documents/OneDrive/Documents/2020/Liposomes Vitamins/LiposomeFormulation"
save_path = r"/Users/calvin/Documents/OneDrive/Documents/2022/Data_Output"

os.chdir(wrk_path_3)
results_path = os.path.join(save_path,"Results_Complete.csv")
dataframe = pd.read_csv(results_path)

print(dataframe.head())

'''
1. Create list of component 1
2. Create list of component 2
3. Create list of component 3
4. Create list of component 4

1. Create list of different concentrations
2. Create list of different dispense speeds
3. Create list of different temperatures (Maybe not for now)

'''

'''Creating Component 1 dataframe
1. HSPC
2. POPC
3. DPPC

this is will attach to dataframe['Component_1']

&

Creating Ratio of Vol by Vol for Component 1 DF
this attached to dataframe['Ratio_1']
'''

component_1_df = pd.DataFrame(columns=['component_1'])
list_component_lipids = ['HSPC',
                         'POPC',
                         'DPPC']
component_1_df['component_1'] = list_component_lipids
print(component_1_df)

vol_vol_pcnt_1_df = pd.DataFrame(columns=['vol_vol_pcnt_1'])
vol_vol_pcnt = [0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05]
vol_vol_pcnt_1_df['vol_vol_pcnt_1'] = [i for i in vol_vol_pcnt if i >= 0.50]
print(vol_vol_pcnt_1_df)
'''Creating Component 2 dataframe (Drugs/Vitamins)
1. Vitamin E
2. Vitamin D
3. Cholesterol

Need to expand list

this will attach to dataframe['Component_2']

&

Creating Ratio of Vol by Vol for Component 2 DF
this attached to dataframe['Ratio_2']

'''

component_2_df = pd.DataFrame(columns=['component_2'])
list_component_drug = ['Vitamin E',
                       'Vitamin D3',
                       'Cholesterol']
component_2_df['component_2'] = list_component_drug
print(component_2_df)

vol_vol_pcnt_2_df = pd.DataFrame(columns=['vol_vol_pcnt_2'])
vol_vol_pcnt_2_df['vol_vol_pcnt_2'] = [i for i in vol_vol_pcnt if i <= 0.50]
print(vol_vol_pcnt_2_df)

'''
Creating Component 3 DataFrame

At this point in time it is None|0
'''
component_3_df = pd.DataFrame(columns=['component_3'])
list_of_component_drug_2 = ['None']
component_3_df['component_3'] = list_of_component_drug_2
print(component_3_df)

vol_vol_pcnt_3_df = pd.DataFrame(columns=['vol_vol_pcnt_3'])
vol_vol_pcnt_3_df['vol_vol_pcnt_3'] = [0]
'''Creating Component 4 DataFrame (Stealth Polymer)
1. PEG - 2000

this will attach to dataframe['Component_4'] - Doesn't exist, need to bring it back in, then remove
Creating Ratio of Vol by Vol for Component 4 DF
this attached to dataframe['Ratio_4']
'''

component_4_df = pd.DataFrame(columns=['component_4'])
list_component_polymer = ['PEG2000 DSPE']
component_4_df['component_4'] = list_component_polymer
print(component_4_df)

vol_vol_pcnt_4_df = pd.DataFrame(columns=['vol_vol_pcnt_4'])
vol_vol_pcnt_4_df['vol_vol_pcnt_4'] = [i for i in vol_vol_pcnt if i <= 0.10]

'''
Creating the concentration DataFrames for each component
The chosen concentrations(mM):
1. 100
2. 80
3. 60
4. 50
5. 30
6. 10
7. 200
'''

conc = [100,
        80,
        60,
        50,
        30,
        10,
        200]

conc_1_df = pd.DataFrame(columns=['Concentration_1 (mM)'])
conc_2_df = pd.DataFrame(columns=['Overall_Concentration_2'])
conc_3_df = pd.DataFrame(columns=['Overall_Concentration_3'])
conc_4_df = pd.DataFrame(columns=['Overall_Concentration_4'])

conc_1_df['Concentration_1 (mM)'] = conc
conc_2_df['Overall_Concetration_2'] = conc
conc_3_df['Overall_Concetration_3'] = [0]
conc_4_df['Overall_Concentration_4'] = conc

'''
Creating the dispense speed dataframe
While the speed can be anywhere between 1-400ul/s, only 3 speeds shall be chosen to reduce the number of experiments required
'''
# speed = [400,300,200,120,80,30]
speed = [400, 120, 30]
speed_disp_df = pd.DataFrame(columns=['speed_disp'])
speed_disp_df['speed_disp'] = speed

'''-------------'''
ethanol_list = [0.5,

                1.4,
                1.0]
ethanol_df = pd.DataFrame(columns=['Ethanol'])
ethanol_df['Ethanol'] = ethanol_list

lipid_vol_pcnt = [0.075,
                  0.15]

lipid_vol_pcnt_df = pd.DataFrame(columns=['lipid_vol_pcnt'])
lipid_vol_pcnt_df['lipid_vol_pcnt'] = lipid_vol_pcnt

Final_Vol = [1.3,
             1.4]

final_vol_df = pd.DataFrame(columns=['final_vol'])
final_vol_df['final_vol'] = Final_Vol
print('------------------------')
print('Creating DF Formulation')
print('------------------------')

formulation_combination_df = pd.DataFrame(list(itertools.product(component_1_df.component_1,
                                                                 ethanol_df.Ethanol,
                                                                 component_2_df.component_2,
                                                                 component_3_df.component_3,
                                                                 component_4_df.component_4,
                                                                 vol_vol_pcnt_1_df.vol_vol_pcnt_1,
                                                                 vol_vol_pcnt_2_df.vol_vol_pcnt_2,
                                                                 vol_vol_pcnt_3_df.vol_vol_pcnt_3,
                                                                 vol_vol_pcnt_4_df.vol_vol_pcnt_4,
                                                                 conc_1_df['Concentration_1 (mM)'],
                                                                 conc_2_df['Overall_Concetration_2'],
                                                                 conc_3_df['Overall_Concetration_3'],
                                                                 conc_4_df['Overall_Concentration_4'],
                                                                 speed_disp_df.speed_disp,
                                                                 lipid_vol_pcnt_df.lipid_vol_pcnt,
                                                                 final_vol_df.final_vol
                                                                 )),
                                          columns=['Component_1',
                                                   'Ethanol_1',
                                                   'Component_2',
                                                   'Component_3',
                                                   'Component_4',
                                                   'Ratio_1',
                                                   'Ratio_2',
                                                   'Ratio_3',
                                                   'Ratio_4',
                                                   'Concentration_1 (mM)',
                                                   'Overall_Concentration_2',
                                                   'Overall_Concentration_3',
                                                   'Concentration_4',
                                                   'Dispense_Speed_uls',
                                                   'Lipid_Vol_Pcnt',
                                                   'Final_Vol'])

print(formulation_combination_df)
print(formulation_combination_df.info())

'''
Cleaning Ratio Columns

Total vol by vol percent column
The three vol by vol percent columns are to be summed. 
Then totals that are higher or lower than 100 must be removed 
(as it is a percent and you cannot go over 100, nor should the percent be lower than 100)
'''

col_list = list(formulation_combination_df)
rmve = ('Component_1',
        'Ethanol_1',
        'Component_2',
        'Component_4',
        'Concentration_1 (mM)',
        'Overall_Concentration_2',
        'Concentration_4',
        'Dispense_Speed_uls',
        'Lipid_Vol_Pcnt',
        'Final_Vol')
for i in range(len(rmve)):
    col_list.remove(rmve[i])

formulation_combination_df['total_pcnt'] = formulation_combination_df[col_list].sum(axis=1)
print(formulation_combination_df)
formulation_combination_df = formulation_combination_df.loc[
    formulation_combination_df['total_pcnt'] == 1.00].reset_index(drop=True)
formulation_combination_df.info()

''' Columns I need to attach to 

['Component_1', 
'Concentration_1 (mM)', 
'Req_Weight_1', 
'Ethanol_1',
       'Ratio_1', 
       'Component_2', 
       'Overall_Concentration_2', 
       'Req_Weight_2',
       'Ethanol_2', 
       'Ratio_2', 
       'Component_3', 
       'Overall_Concentration_3',
       'Req_Weight_3', 
       'Ratio_3', 
       'Concentration_4', 
       'Req_Weight_4',
       'Ethanol_4', 
       'Ratio_4', 
       'Final_Vol', 
       'Lipid_Vol_Pcnt',
       'Dispense_Speed_uls', 
       'final_lipid_volume', 
       'component_1_vol',
       'component_2_vol', 
       'component_3_vol', 
       'component_4_vol',
       'component_1_vol_conc', 
       'component_1_vol_stock', 
       'component_2_vol_conc',
       'component_2_vol_stock', 
       'component_3_vol_conc',
       'component_3_vol_stock', 
       'ethanol_dil', 
       'Final_Concentration',
       'ES_Aggregation', 
       'Duplicate_Check', 
       'Z-Average (d.nm)', 
       'PdI',
       'PdI Width (d.nm)', 
       'xlogp_cp_1', 
       'complexity_cp_1',
       'heavy_atom_count_cp_1', 
       'single_bond_cp_1', 
       'double_bond_cp_1',
       'xlogp_cp_2', 
       'complexity_cp_2', 
       'heavy_atom_count_cp_2', 
       'tpsa_cp_2',
       'ssr_cp_2', 
       'single_bond_cp_2', 
       'double_bond_cp_2',
       'aromatic_bond_cp_2'],

'''

'''Let's create a copy of the formulation combination'''

unlabelled_df = formulation_combination_df.copy()

'''---------'''
pubchem_path = os.path.join(save_path, "pubchem_data.csv")
pubchem_df = pd.read_csv(pubchem_path)

unlabelled_df_merge = pd.merge(unlabelled_df, pubchem_df,
                               how="inner",
                               left_on="Component_1",
                               right_on="compound",
                               suffixes=["_cp_1"]).reset_index(drop=True)
unlabelled_df_merge = pd.merge(unlabelled_df_merge, pubchem_df,
                               how="inner",
                               left_on="Component_2",
                               right_on="compound",
                               suffixes=["_cp_1", "_cp_2"]).reset_index(drop=True)
temp_cp3 = pubchem_df.copy()
temp_cp3.columns += "_cp_3"

unlabelled_df_merge = pd.merge(unlabelled_df_merge, temp_cp3,
                                 how="inner",
                                 left_on="Component_3",
                                 right_on="compound_cp_3",
                                 suffixes=["_cp_1", "_cp_2", "_cp_3"]).reset_index(drop=True)
temp_cp4 = pubchem_df.copy()
temp_cp4.columns += "_cp_4"
unlabelled_df_merge = pd.merge(unlabelled_df_merge, temp_cp4,
                                   how="inner",
                                   left_on="Component_4",
                                   right_on="compound_cp_4",
                                   ).reset_index(drop=True)
'''------'''

unlabelled_df_merge['Ethanol_2'] = unlabelled_df_merge['Ethanol_1']
unlabelled_df_merge['Ethanol_4'] = unlabelled_df_merge['Ethanol_1']

'''Get Required Weight'''
unlabelled_df_merge['Req_Weight_1'] = ((unlabelled_df_merge['Concentration_1 (mM)'].astype(float) * (10 ** (-3))) * (
            unlabelled_df_merge['Ethanol_1'].astype(float) * (10 ** (-3)))) \
                                      * (unlabelled_df_merge['mw_cp_1'])

unlabelled_df_merge['Req_Weight_2'] = ((unlabelled_df_merge['Overall_Concentration_2'].astype(float) * (10 ** (-3))) * (
            unlabelled_df_merge['Ethanol_2'].astype(float) * (10 ** (-3)))) \
                                      * (unlabelled_df_merge['mw_cp_2'])

unlabelled_df_merge['Req_Weight_3'] = 0
unlabelled_df_merge['Overall_Concentration_3'] = 0
unlabelled_df_merge['Ratio_3'] = 0
# TODO Change the req_weight_3 later and overall_concentration_3

mw_peg2000 = float(2790.52)

unlabelled_df_merge['Req_Weight_4'] = ((unlabelled_df_merge['Concentration_4'].astype(float) * (10 ** (-3))) * (
            unlabelled_df_merge['Ethanol_4'].astype(float) * (10 ** (-3)))) \
                                      * mw_peg2000

'''final_lipid_vol'''
unlabelled_df_merge['final_lipid_volume'] = unlabelled_df_merge['Lipid_Vol_Pcnt'] * unlabelled_df_merge['Final_Vol']  \
                                            * 1000
'''component_1_vol'''
unlabelled_df_merge['component_1_vol'] = (unlabelled_df_merge['Ratio_1'] * unlabelled_df_merge['final_lipid_volume'])
unlabelled_df_merge['component_2_vol'] = (unlabelled_df_merge['Ratio_2'] * unlabelled_df_merge['final_lipid_volume'])
unlabelled_df_merge['component_3_vol'] = (unlabelled_df_merge['Ratio_3'] * unlabelled_df_merge['final_lipid_volume'])
unlabelled_df_merge['component_4_vol'] = (unlabelled_df_merge['Ratio_4'] * unlabelled_df_merge['final_lipid_volume'])

unlabelled_df_merge['component_1_vol_conc'] = unlabelled_df_merge['Concentration_1 (mM)']
unlabelled_df_merge['component_1_vol_stock'] = unlabelled_df_merge['component_1_vol'] * unlabelled_df_merge[
    'component_1_vol_conc'] / unlabelled_df_merge['Concentration_1 (mM)']

unlabelled_df_merge['component_2_vol_conc'] = unlabelled_df_merge['Overall_Concentration_2']
unlabelled_df_merge['component_2_vol_stock'] = unlabelled_df_merge['component_2_vol'] * unlabelled_df_merge[
    'component_2_vol_conc'] / unlabelled_df_merge['Overall_Concentration_2']

unlabelled_df_merge['component_3_vol_conc'] = unlabelled_df_merge['Overall_Concentration_3']
unlabelled_df_merge['component_3_vol_stock'] = unlabelled_df_merge['component_3_vol'] * unlabelled_df_merge[
    'component_3_vol_conc'] / unlabelled_df_merge['Overall_Concentration_3']
unlabelled_df_merge['component_3_vol_stock'] = unlabelled_df_merge['component_3_vol_stock'].where(
    unlabelled_df_merge['component_3_vol_stock'].isnull(), 1).fillna(0).astype(int)

unlabelled_df_merge['ethanol_dil'] = (unlabelled_df_merge['component_1_vol'] + \
                                      unlabelled_df_merge['component_2_vol'] + \
                                      unlabelled_df_merge['component_3_vol']) - \
                                     (unlabelled_df_merge['component_1_vol_stock'] + \
                                      unlabelled_df_merge['component_2_vol_stock'] + \
                                      unlabelled_df_merge['component_3_vol_stock'])
# TODO Need to create varying component volumes so that dilution is required

unlabelled_df_merge['Final_Concentration'] = ((unlabelled_df_merge['Ratio_1'] * unlabelled_df_merge["component_1_vol_conc"] * (
                unlabelled_df_merge['final_lipid_volume'] / 1000)) + (unlabelled_df_merge['Ratio_2'] * unlabelled_df_merge['component_2_vol_conc'] * (
                unlabelled_df_merge['final_lipid_volume'] / 1000)) + (unlabelled_df_merge['Ratio_3'] * unlabelled_df_merge['component_3_vol_conc'] * (
                unlabelled_df_merge['final_lipid_volume'] / 1000)) + (unlabelled_df_merge['Ratio_4'] * unlabelled_df_merge['Concentration_4'] * (
                unlabelled_df_merge['final_lipid_volume'] / 1000))) / (unlabelled_df_merge['Final_Vol'])

'''Need this for the end'''

columns = ['ES_Aggregation',
           'Duplicate_Check',
           'Z-Average (d.nm)',
           'PdI',
           'PdI Width (d.nm)']
unlabelled_df_merge[columns] = np.nan

unlabelled_df_merge.drop(columns=['compound_cp_1', 'compound_cp_2', 'cid_cp_1', 'cid_cp_2',
                                          'compound_cp_3', 'compound_cp_4', 'cid_cp_3', 'cid_cp_4'],
                         # 'mw_cp_1','mw_cp_2'],
                         inplace=True)

##Compare
col_not_unique = []
for col in unlabelled_df_merge.columns:
    if (len(unlabelled_df_merge[col].unique()) == 1) and (unlabelled_df_merge[col].isnull().all() == False) \
            and ((unlabelled_df_merge[col].all() == 0) == False)\
            and (col != 'h_bond_donor_count_cp_2') == True:
        col_not_unique.append(str(col))
        unlabelled_df_merge.drop(col, inplace=True, axis=1)

# Drop anyway

unlabelled_df_merge.drop(columns=['aromatic_bond_cp_1'], inplace=True)

unlabelled_df_reorder = unlabelled_df_merge.reindex(columns=dataframe.columns)

df_compare = [dataframe, unlabelled_df_reorder]

all([len(df_compare[0].columns.intersection(df.columns))
     == df_compare[0].shape[1] for df in df_compare])

'''
Need to remove volume values less than 3 ml due to the robot not being to handle it
'''
unlabelled_df_reorder = unlabelled_df_reorder[
    (unlabelled_df_reorder.component_1_vol > 3) | (unlabelled_df_reorder.component_2_vol > 3) | (
                unlabelled_df_reorder.component_3_vol > 3) | (unlabelled_df_reorder.component_4_vol > 3)]

os.chdir(save_path)
unlabelled_df_reorder.to_csv('unlabelled_data_full.csv', index=False)


def stratified_sample(df, strata, size=None, seed=None, keep_index=True):
    '''
    It samples data from a pandas dataframe using strata. These functions use
    proportionate stratification:
    n1 = (N1/N) * n
    where:
        - n1 is the sample size of stratum 1
        - N1 is the population size of stratum 1
        - N is the total population size
        - n is the sampling size
    Parameters
    ----------
    :df: pandas dataframe from which data will be sampled.
    :strata: list containing columns that will be used in the stratified sampling.
    :size: sampling size. If not informed, a sampling size will be calculated
        using Cochran adjusted sampling formula:
        cochran_n = (Z**2 * p * q) /e**2
        where:
            - Z is the z-value. In this case we use 1.96 representing 95%
            - p is the estimated proportion of the population which has an
                attribute. In this case we use 0.5
            - q is 1-p
            - e is the margin of error
        This formula is adjusted as follows:
        adjusted_cochran = cochran_n / 1+((cochran_n -1)/N)
        where:
            - cochran_n = result of the previous formula
            - N is the population size
    :seed: sampling seed
    :keep_index: if True, it keeps a column with the original population index indicator

    Returns
    -------
    A sampled pandas dataframe based in a set of strata.
    Examples
    --------
    >> df.head()
        id  sex age city
    0   123 M   20  XYZ
    1   456 M   25  XYZ
    2   789 M   21  YZX
    3   987 F   40  ZXY
    4   654 M   45  ZXY
    ...
    # This returns a sample stratified by sex and city containing 30% of the size of
    # the original data
    >> stratified = stratified_sample(df=df, strata=['sex', 'city'], size=0.3)
    Requirements
    ------------
    - pandas
    - numpy
    '''
    population = len(df)
    size = __smpl_size(population, size)
    tmp = df[strata]
    tmp['size'] = 1
    tmp_grpd = tmp.groupby(strata).count().reset_index()
    tmp_grpd['samp_size'] = round(size / population * tmp_grpd['size']).astype(int)

    # controlling variable to create the dataframe or append to it
    first = True
    for i in range(len(tmp_grpd)):
        # query generator for each iteration
        qry = ''
        for s in range(len(strata)):
            stratum = strata[s]
            value = tmp_grpd.iloc[i][stratum]
            n = tmp_grpd.iloc[i]['samp_size']

            if type(value) == str:
                value = "'" + str(value) + "'"

            if s != len(strata) - 1:
                qry = qry + stratum + ' == ' + str(value) + ' & '
            else:
                qry = qry + stratum + ' == ' + str(value)

        # final dataframe
        if first:
            stratified_df = df.query(qry).sample(n=n, random_state=seed).reset_index(drop=(not keep_index))
            first = False
        else:
            tmp_df = df.query(qry).sample(n=n, random_state=seed).reset_index(drop=(not keep_index))
            stratified_df = stratified_df.append(tmp_df, ignore_index=True)

    return stratified_df


def stratified_sample_report(df, strata, size=None):
    '''
    Generates a dataframe reporting the counts in each stratum and the counts
    for the final sampled dataframe.
    Parameters
    ----------
    :df: pandas dataframe from which data will be sampled.
    :strata: list containing columns that will be used in the stratified sampling.
    :size: sampling size. If not informed, a sampling size will be calculated
        using Cochran adjusted sampling formula:
        cochran_n = (Z**2 * p * q) /e**2
        where:
            - Z is the z-value. In this case we use 1.96 representing 95%
            - p is the estimated proportion of the population which has an
                attribute. In this case we use 0.5
            - q is 1-p
            - e is the margin of error
        This formula is adjusted as follows:
        adjusted_cochran = cochran_n / 1+((cochran_n -1)/N)
        where:
            - cochran_n = result of the previous formula
            - N is the population size
    Returns
    -------
    A dataframe reporting the counts in each stratum and the counts
    for the final sampled dataframe.
    '''
    population = len(df)
    size = __smpl_size(population, size)
    tmp = df[strata]
    tmp['size'] = 1
    tmp_grpd = tmp.groupby(strata).count().reset_index()
    tmp_grpd['samp_size'] = round(size / population * tmp_grpd['size']).astype(int)
    return tmp_grpd


def __smpl_size(population, size):
    '''
    A function to compute the sample size. If not informed, a sampling
    size will be calculated using Cochran adjusted sampling formula:
        cochran_n = (Z**2 * p * q) /e**2
        where:
            - Z is the z-value. In this case we use 1.96 representing 95%
            - p is the estimated proportion of the population which has an
                attribute. In this case we use 0.5
            - q is 1-p
            - e is the margin of error
        This formula is adjusted as follows:
        adjusted_cochran = cochran_n / 1+((cochran_n -1)/N)
        where:
            - cochran_n = result of the previous formula
            - N is the population size
    Parameters
    ----------
        :population: population size
        :size: sample size (default = None)
    Returns
    -------
    Calculated sample size to be used in the functions:
        - stratified_sample
        - stratified_sample_report
    '''
    if size is None:
        cochran_n = round(((1.96) ** 2 * 0.5 * 0.5) / 0.02 ** 2)
        n = round(cochran_n / (1 + ((cochran_n - 1) / population)))
    elif size >= 0 and size < 1:
        n = round(population * size)
    elif size < 0:
        raise ValueError('Parameter "size" must be an integer or a proportion between 0 and 0.99.')
    elif size >= 1:
        n = size
    return n


stratified_sample_report(unlabelled_df_reorder, ['Component_1', 'Component_2'])

sample_df = stratified_sample(unlabelled_df_reorder, ['Component_1', 'Component_2'], size=10000, seed=123,
                              keep_index=False)
sample_df

sample_df.to_csv('stratified_sample_experiment.csv', index=False)
