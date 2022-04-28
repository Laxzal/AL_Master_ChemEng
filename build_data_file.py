import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import pandas as pd
from scipy.stats import zscore
import pubchempy as pcp
from rdkit import Chem

'''
Changing Directory
'''

wrk_path_1 = r"C:\Users\calvi\OneDrive\Documents\2020\Liposomes Vitamins\LiposomeFormulation"
wrk_path_2 = r"C:\Users\Calvin\OneDrive\Documents\2020\Liposomes Vitamins\LiposomeFormulation"
wrk_path_3 = r"/Users/calvin/Documents/OneDrive/Documents/2020/Liposomes Vitamins/LiposomeFormulation"

data_output_path_1 = r"/Users/calvin/Documents/OneDrive/Documents/2022/Data_Output"
data_output_path_2 = r"C:\Users\Calvin\OneDrive\Documents\2022\Data_Output"

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

'''
Data Files
File Names:

Set the three file names as variables. Files required:
Formulation Master (FM) Excel
Aggregation Result CSVs
DLS Files
'''

master_file = r"LiposomeMasterFile_Fixed.xlsm"
results_file = r"Results.xlsx"
path_dls = r"/Users/calvin/Documents/OneDrive/Documents/2020/Liposomes Vitamins/DLS Data/GUID Versions"

'''
Extracting Data
FM Sheets

The FM excel file contains individual sheets with the specific formulations used. They are to be extracted into one 
DataFrame. There are 4 Sheets that need to be ignored:

Sheet1
Data
Template
Component MWT
'''


class DataBuild:

    def __init__(self, master_file: str=r"LiposomeMasterFile_Fixed.xlsm", results_file: str= r"Results.xlsx"):
        self.extracted_sheets = []
        self.ignore_sheets = ['Sheet1', 'Data', 'Template', 'Component MWT']
        self.master_file = master_file
        self.results_file = results_file
        self.data_sheet = None
        self.formulation_sheet = None
        self.path_dls = r"/Users/calvin/Documents/OneDrive/Documents/2020/Liposomes Vitamins/DLS Data/GUID Versions"
        self.files = []
        self.dls_data = pd.DataFrame()

    def read_in_data_sheets(self):

        tmp_df = pd.read_excel(self.master_file, sheet_name=None)
        df = pd.DataFrame()
        for key, value in tmp_df.items():
            if key in self.ignore_sheets:
                continue
            self.extracted_sheets.append(key)
            df = df.append(value, ignore_index=True)

        df["Sample"] = df['Experiment'].str.replace(r"(_)([A-Z])(\d*)(?=\b)$", r"", regex=True)
        df.replace(np.nan, 0, inplace=True)
        self.data_sheet = df

        return self.data_sheet

    def read_formulation_sheet(self):
        data_df = pd.read_excel(self.master_file, sheet_name='Data', skiprows=[0])
        data_df.replace(np.nan, 0, inplace=True)
        self.formulation_sheet = data_df

        return self.formulation_sheet

    def merge_sheets_and_data_check(self):
        merge_df = pd.merge(self.formulation_sheet, self.data_sheet, how="outer", left_on="GUID",
                            right_on="Sample")

        merge_fail_guid = np.where(pd.isnull(merge_df['GUID']))
        merge_fail_sample = np.where(pd.isnull(merge_df['Sample']))
        print("Join failure on GUID:")
        print(merge_fail_guid)
        print("---------------------------------------------")
        print("Join failure on Sample:")
        print(merge_fail_sample)

        if len(merge_fail_guid) != 0:
            for i in merge_fail_guid:
                print("Information on GUID join fails")
                print(merge_df.iloc[i])

        if len(merge_fail_sample) != 0:
            for i in merge_fail_sample:
                print("Information on Sample join fails")
                print(merge_df.iloc[i])

        # Final Concentration Add

        merge_df['Final_Concentration'] = ((merge_df['Ratio_1'] * merge_df["component_1_vol_conc"] * (
                merge_df['final_lipid_volume'] / 1000)) + (merge_df['Ratio_2'] * merge_df['component_2_vol_conc'] * (
                merge_df['final_lipid_volume'] / 1000)) + (merge_df['Ratio_3'] * merge_df['component_3_vol_conc'] * (
                merge_df['final_lipid_volume'] / 1000)) + (merge_df['Ratio_4'] * merge_df['Concentration_4'] * (
                merge_df['final_lipid_volume'] / 1000))) / (merge_df['Final_Vol'])

        print("Checking for NaNs...")
        print(merge_df.isnull().values.any())

        self.formulation_df = merge_df

        return self.formulation_df

    def read_results_file(self):
        results_df = pd.read_excel(self.results_file)
        print(results_df.shape)

        # GUID & Experiment need to be combined for the joining process as well as removing unnecessary columns
        results_df['experiment_comb'] = results_df['GUID'] + str("_") + results_df['Experiment']
        results_df.drop(columns=['Experiment', 'GUID'], inplace=True)
        print(results_df.head())

        self.results_df = results_df

        return self.results_df

    def merge_formu_results(self):
        merge_results_df = pd.merge(self.formulation_df, self.results_df, how="outer", left_on="Experiment",
                                    right_on="experiment_comb").reset_index(drop=True)
        print("Rows with NaN values...")
        print(merge_results_df[merge_results_df.isnull().any(axis=1)])

        self.formulation_results = merge_results_df

        return self.formulation_results

    def missing_aggregation_check(self):
        missing_aggregation_df = self.formulation_results[self.formulation_results['experiment_comb'].isna()]
        self.list_missing_aggregation = list(missing_aggregation_df['GUID'].unique())
        print("Number of total possible experiments that have not had their aggregation recorded",
              len(missing_aggregation_df['GUID']))
        print("Number of unique experiments that have not had their aggregation recorded: ",
              len(self.list_missing_aggregation))

    def extract_dls_data(self):
        for (dirpath, dirnames, filenames) in os.walk(self.path_dls):
            self.files.extend(filenames)

        # print(files)
        for f in self.files:
            if f.endswith(('.csv', '.CSV')):
                print(str(f))
                dw = pd.read_csv(os.path.join(self.path_dls, f), encoding="ISO-8859-8")
                self.dls_data = self.dls_data.append(dw, ignore_index=True)


        return self.dls_data

    def clean_dls_data(self):
        self.dls_data['Sample Name'] = self.dls_data['Sample Name'].str.strip()  # Clean up white spaces
        self.dls_data['Sample Name'] = self.dls_data['Sample Name'].str.replace("(?=)(\s)(\d)", "",
                                                                                regex=True)  # Remove the numbers proceeding the D(value)
        self.dls_data['Sample Name'] = self.dls_data['Sample Name'].str.replace("(\s)(?<=)(\-)(?=)(\s)", "_",
                                                                                regex=True)  # Change the - into an _
        self.dls_data['Sample Name'] = self.dls_data['Sample Name'].str.replace("(?<=[A-Z]|\d)(\s)(?=\D)", "_",
                                                                                regex=True)  # Put an underscore between the GUID and D(value)
        self.dls_data['Sample Name'] = self.dls_data['Sample Name'].str.replace("(_FILTERED)", "",
                                                                                regex=True)  # Unique instance of putting the
        # word filtered in dls naming
        self.dls_data['Sample Name'] = self.dls_data['Sample Name'].str.replace("(_FILTERED\d+)", "", regex=True)
        self.dls_data['Sample Name'] = self.dls_data['Sample Name'].str.strip()  # For good measure

        self.list_dls_unique_scans = list(self.dls_data['Sample Name'].unique())
        print("Number of unique DLS scans:", self.dls_data['Sample Name'].nunique())
        print("DLS Samples Scanned: ", *iter(self.list_dls_unique_scans), sep=' | ')

    def z_scoring(self, threshold: float = 1.0):
        '''
        Create a DataFrame with the Z-Average (d.nm) [Average], PDI [Average] and PDI Width [Average]. Z score is
        used to determine whether the DLS iterration scan values are within the threshold. If not, they are removed
        and the mean is once again determined. A mean vs std is utilised when determining whether to use the Z score.
        A Standard Deviation should not be larger than the Mean. Please note that the filtered out block of code is
        the original code that does not take into account the Z score. If deemed better for results, then utilise/
        PdI Width may be useless and there is argument to also bring in the volume and count values
        :return:
        '''

        columns = [col for col in self.dls_data.columns if
                   'Z-Average' in col or 'Sample Name' in col or 'PdI' in col or 'PdI Width' in col]
        dlsdata_average = pd.DataFrame(columns=columns)

        z_score_filter_df = pd.DataFrame(columns=['Sample Name', 'Z-Average', 'PdI', 'PdI_Width'])

        for index in self.list_dls_unique_scans:
            temp_df = self.dls_data[self.dls_data['Sample Name'] == (index)]
            temp_df = temp_df[['Z-Average (d.nm)', 'PdI', 'PdI Width (d.nm)']]
            # create variable for checking
            location_s = None
            location_pdi = None
            location_pdi_width = None
            # Z-Average
            temp_df_mean = temp_df['Z-Average (d.nm)'].mean()
            temp_df_std = temp_df['Z-Average (d.nm)'].std()
            if (temp_df_mean < temp_df_std) == True:
                z_s = np.abs(zscore(temp_df['Z-Average (d.nm)']))
                location_s = np.where(z_s > threshold)
                for i in range(len(location_s[0])):
                    print("Dropping size: ", len(location_s[0]), str(index))
                    temp_df_s = temp_df.reset_index(drop=True).drop([location_s[0][i]])
                temp_df_mean = temp_df_s['Z-Average (d.nm)'].mean()

            if location_s is not None:
                z_ave_cnt = len(location_s[0])
            else:
                z_ave_cnt = 0

            # PdI
            temp_df_mean_p = temp_df['PdI'].mean()
            temp_df_std_p = temp_df['PdI'].std()
            if (temp_df_mean_p < temp_df_std_p) == True:
                z_pdi = np.abs(zscore(temp_df['PdI']))
                location_pdi = np.where(z_pdi > threshold)
                for i in range(len(location_pdi[0])):
                    print("Dropping PdI: ", len(location_pdi[0]), str(index))
                    temp_df_s = temp_df.reset_index(drop=True).drop([location_pdi[0][i]])
                temp_df_mean_p = temp_df_s['PdI'].mean()

            if location_pdi is not None:
                pdi_ave_cnt = len(location_pdi[0])
            else:
                pdi_ave_cnt = 0

            # PdI Width
            temp_df_mean_pw = temp_df['PdI Width (d.nm)'].mean()
            temp_df_std_pw = temp_df['PdI Width (d.nm)'].std()
            if (temp_df_mean_pw < temp_df_std_pw) == True:
                z_pdi_width = np.abs(zscore(temp_df['PdI Width (d.nm)']))
                location_pdi_width = np.where(z_pdi_width > threshold)
                for i in range(len(location_pdi_width[0])):
                    print("Dropping PdI Width: ", len(location_pdi_width[0]), str(index))
                    temp_df_s = temp_df.reset_index(drop=True).drop([location_pdi_width[0][i]])
                temp_df_mean_pw = temp_df_s['PdI Width (d.nm)'].mean()

            if location_pdi_width is not None:
                pdiw_ave_cnt = len(location_pdi_width[0])
            else:
                pdiw_ave_cnt = 0

            z_score_filter_df = z_score_filter_df.append({'Sample Name': str(index),
                                                          'Z-Average': int(z_ave_cnt),
                                                          'PdI': int(pdi_ave_cnt),
                                                          'PdI_Width': int(pdiw_ave_cnt)}, ignore_index=True)

            dlsdata_average = dlsdata_average.append({'Sample Name': str(index),
                                                      'Z-Average (d.nm)': float(temp_df_mean),
                                                      'PdI': float(temp_df_mean_p),
                                                      'PdI Width (d.nm)': float(temp_df_mean_pw)}, ignore_index=True)

        print(z_score_filter_df)
        self.dlsdata_average = dlsdata_average

    def merge_form_results_dls(self):
        # Need to remove bad joins where the GUID is empty and the sample Name,
        # essentially the field was not appropriate enough for comple join
        # Creating this list will then be used to join on another column
        # TODO To be reworked because the list remove logic is wrong
        results_comp_df = pd.merge(self.formulation_results, self.dlsdata_average,
                                   how="outer",
                                   left_on="experiment_comb",
                                   right_on="Sample Name").reset_index(drop=True)

        list_remove = []
        for index, values in results_comp_df.iterrows():
            if pd.isna(values['GUID']) == True and pd.isna(values['Sample Name']) == True:
                list_remove.append(values['Sample Name'])

        print("List of items to be removed")

        print(list_remove)
        merge_dlsdata = self.dlsdata_average[~self.dlsdata_average['Sample Name'].isin(list_remove)]

        results_comp_df = pd.merge(results_comp_df, merge_dlsdata,
                                   how="outer",
                                   left_on="Experiment",
                                   right_on="Sample Name").reset_index(drop=True)
        '''Since duplicate columns were created during the two merges, they need to be combined. A where statement is 
        used to determine if the original row is null, and the duplicate row is, then the duplicate row value is 
        copied into the original row. '''
        results_comp_df['Sample Name'] = results_comp_df['Sample Name_x'].where(
            results_comp_df['Sample Name_x'].notnull(), results_comp_df['Sample Name_y'])
        results_comp_df['Z-Average (d.nm)'] = results_comp_df['Z-Average (d.nm)_x'].where(
            results_comp_df['Z-Average (d.nm)_x'].notnull(), results_comp_df['Z-Average (d.nm)_y'])
        results_comp_df['PdI'] = results_comp_df['PdI_x'].where(results_comp_df['PdI_x'].notnull(),
                                                                results_comp_df['PdI_y'])
        results_comp_df['PdI Width (d.nm)'] = results_comp_df['PdI Width (d.nm)_x'].where(
            results_comp_df['PdI Width (d.nm)_x'].notnull(), results_comp_df['PdI Width (d.nm)_y'])
        # Columns that are now redundant are removed
        clms_drop = ['Sample Name_x', 'Sample Name_y',
                     'Z-Average (d.nm)_x', 'Z-Average (d.nm)_y',
                     'PdI_x', 'PdI_y',
                     'PdI Width (d.nm)_x', 'PdI Width (d.nm)_y']
        results_comp_df.drop(columns=clms_drop, inplace=True)

        print('Displaying columns to ensure that duplicate columns have been removed')
        print(results_comp_df.columns)
        self.master_formulation_results_df = results_comp_df

    def print_master_formulation_results_info(self):

        print(self.master_formulation_results_df.info())

    def data_size_aggregation_check(self):
        list_missing_size = []
        for index, values in self.master_formulation_results_df.iterrows():
            if pd.isna(values['GUID']) == False and pd.isna(values['Z-Average (d.nm)']):
                list_missing_size.append(values['GUID'])
        list_missing_size = list(set(list_missing_size))
        print(list_missing_size)
        print("Number GUIDs that do not have Particle Average Size:", len(list_missing_size))

        self.missing_sizes_guid = pd.DataFrame(list_missing_size)


        self.extract = self.master_formulation_results_df[self.master_formulation_results_df['GUID'].isin(list_missing_size)]


        list_missing_aggregation = []
        for index, values in self.master_formulation_results_df.iterrows():
            if pd.isna(values['GUID']) == False and pd.isna(values['ES_Aggregation']) == True:
                list_missing_aggregation.append(values['GUID'])

        list_missing_aggregation = list(set(list_missing_aggregation))
        print(list_missing_aggregation)
        print("Number GUIDs that do not have an Aggregation Check:", len(list_missing_aggregation))

        # How many formulations were there attempted (incl. missing sizes/aggregation)
        list_completed_formulations = list(self.master_formulation_results_df['GUID'].unique())
        print(list_completed_formulations)
        print("Count of completed formulations (incl. missing sizes/aggregation): ", len(list_completed_formulations))

    def cleaning_master_df_(self):
        '''
      Complete general cleaning and then split the files.

        General:

        Need to ensure that the Component Rows that are written as 0 are changed to None
        Remove columns that are not required, such as GUID, Conc. Ranges, Comments, Sample Names, Links, Dates etc.
        Remove columns that are not unique (Will be added to a list)



      :return:
      '''

        self.master_formulation_results_df['Component_2'] = self.master_formulation_results_df['Component_2'].apply(
            lambda x: str('None') if x == 0 else x)
        self.master_formulation_results_df['Component_3'] = self.master_formulation_results_df['Component_3'].apply(
            lambda x: str('None') if x == 0 else x)
        self.master_formulation_results_df = self.master_formulation_results_df[
            self.master_formulation_results_df['GUID'].notna()].reset_index(drop=True)

        print('Any NaN in GUID column: ', self.master_formulation_results_df.isna().any())

        self.master_formulation_results_df = self.master_formulation_results_df.dropna(axis=0,
                                                                                       subset=['Z-Average (d.nm)',
                                                                                               'ES_Aggregation'],
                                                                                       thresh=1).reset_index(drop=True)
        print('Unique phospholipids: ', self.master_formulation_results_df['Component_1'].unique())

        clms_drop = ['Date', 'GUID', 'Range_Concetration_2',
                     'Range_Concentration_3',
                     'Link', 'Experiment', 'Sample', 'Comments', 'experiment_comb',
                     'Sample Name']
        self.master_formulation_results_df.drop(columns=clms_drop, inplace=True)


        print('Columns: ', self.master_formulation_results_df.columns)
        self.master_formulation_results_df['Component_2'].replace('Vitamin D', 'Vitamin D3', inplace=True)
        self.master_formulation_results_df['Component_3'].replace('Vitamin D', 'Vitamin D3', inplace=True)

    def pubchem_data(self):

        component_pubchem_df = pd.DataFrame(
            columns=['compound', 'cid', 'mw', 'xlogp', 'complexity', 'heavy_atom_count', 'tpsa', 'ssr', 'single_bond',
                     'double_bond', 'aromatic_bond'])
        self.master_components = self.master_formulation_results_df[['Component_1', 'Component_2', 'Component_3']]

        components = pd.Series(self.master_components.values.ravel('F')).unique()

        for item in components:
            if item == 'None':
                continue
            else:
                print(item)
                results_x = pcp.get_compounds(item, 'name')
                # Get smallest set of smallest ring
                print(results_x)
                compound = pcp.Compound.from_cid(results_x[0].cid)
                m = Chem.MolFromSmiles(compound.isomeric_smiles)
                ssr = Chem.GetSymmSSSR(m)
                print(len(ssr))

                # Get Count of bond types
                single_bond = 0
                double_bond = 0
                aromatic_bond = 0
                for b in m.GetBonds():
                    # print(b.GetBondType(), b.GetBondTypeAsDouble(),b.GetIsAromatic() )
                    if b.GetBondType() == 1.0:  # Appears 1.0 is equivalent to single bond
                        single_bond = 1 + single_bond
                    if b.GetBondType() == 2.0:  # Appears 2.0 is equivalent to double bond
                        double_bond = 1 + double_bond
                    if b.GetIsAromatic() == True:
                        aromatic_bond = 1 + aromatic_bond

                        # append to dataframe

                component_pubchem_df = component_pubchem_df.append({'compound': str(item),
                                                                    'cid': int(results_x[0].cid),
                                                                    'mw': float(results_x[0].molecular_weight),
                                                                    'xlogp': float(results_x[0].xlogp),
                                                                    'complexity': int(results_x[0].complexity),
                                                                    'heavy_atom_count': int(
                                                                        results_x[0].heavy_atom_count),
                                                                    'tpsa': int(results_x[0].tpsa),
                                                                    'ssr': int(len(ssr)),
                                                                    'single_bond': int(single_bond),
                                                                    'double_bond': int(double_bond),
                                                                    'aromatic_bond': int(aromatic_bond)},
                                                                   ignore_index=True)
        print(component_pubchem_df)
        self.component_pubchem_df = component_pubchem_df

    def merge_master_pubchem(self):

        results_comp_df_x = pd.merge(self.master_formulation_results_df, self.component_pubchem_df,
                                     how="inner",
                                     left_on="Component_1",
                                     right_on="compound",
                                     suffixes=["_cp_1"]).reset_index(drop=True)
        results_comp_df_x_y = pd.merge(results_comp_df_x, self.component_pubchem_df,
                                       how="inner",
                                       left_on="Component_2",
                                       right_on="compound",
                                       suffixes=["_cp_1", "_cp_2"]).reset_index(drop=True)
        results_comp_df_x_y.drop(columns=['compound_cp_1', 'compound_cp_2', 'cid_cp_1', 'cid_cp_2',
                                 # 'mw_cp_1',
      #                            'mw_cp_2',
     #                               'complexity_cp_2',
      #                              'aromatic_bond_cp_2',
     #                               'double_bond_cp_2',
     #                               'single_bond_cp_2',
     #                               'ssr_cp_2',
     #                               'tpsa_cp_2',
    #                                'heavy_atom_count_cp_2',
   #                                 'xlogp_cp_2',

                                          ],
                                 inplace=True)

        #results_comp_df_x_y.rename(mapper={'mw_cp_1': 'mw_cp',
        #                                   'complexity_cp_1':'complexity',
        #                                   'aromatic_bond_cp_1': 'aromatic_bond',
        #                                   'double_bond_cp_1': 'double_bond',
        #                                   'single_bond_cp_1': 'single_bond',
        #                                   'ssr_cp_1':'ssr',
        #                                   'tpsa_cp_1':'tpsa',
        #                                   'heavy_atom_count_cp_1':'heavy_atom_count',
        #                                   'xlogp_cp_1':'xlogp'},
        #                           axis = 'columns',
        #                           inplace=True)
        self.master_formulation_results_df = results_comp_df_x_y

    def cols_not_unique_check(self):
        col_not_unique = []
        for col in self.master_formulation_results_df.columns:
            if len(self.master_formulation_results_df[col].unique()) == 1:
                col_not_unique.append(str(col))
                self.master_formulation_results_df.drop(col, inplace=True, axis = 1)

    def column_categ_check_object_type(self):
        list_columns = list(self.master_formulation_results_df.columns)
        for i in range(len(list_columns)):
            if self.master_formulation_results_df[list_columns[i]].dtypes == 'O':
                print(list_columns[i])
                print(self.master_formulation_results_df[list_columns[i]].value_counts())

    def output_files(self):

        if os.name == 'posix':
            os.chdir(data_output_path_1)
            print("Utilising MacBook")
        else:
            try:
                os.chdir(data_output_path_2)
                print("Utilising Home Pathway")
            except OSError:
                os.chdir(wrk_path_1)
                print("Utilising Lab Pathway")

        self.master_formulation_results_df.to_csv("Results_Complete.csv", index=False)
        self.formulation_sheet[self.formulation_sheet['GUID'].isin(self.list_missing_aggregation)].to_csv(
            "Rerun_Experiments.csv", index=False)
        self.dls_data.to_csv('dls_data_check.csv')
        self.missing_sizes_guid.to_csv('missing_sizes_guid.csv')
        self.extract.to_csv('missing_sizes_dates.csv')
        self.component_pubchem_df.to_csv('pubchem_data.csv', index=False)

master_file_path = r"LiposomeMasterFile_Fixed.xlsm"
results_file_path = r"Results.xlsx"
dataoutput = DataBuild()
dataoutput.read_in_data_sheets()
dataoutput.read_formulation_sheet()
dataoutput.merge_sheets_and_data_check()
dataoutput.read_results_file()
dataoutput.merge_formu_results()
dataoutput.missing_aggregation_check()
dataoutput.extract_dls_data()
dataoutput.clean_dls_data()
dataoutput.z_scoring(threshold=1.0)
dataoutput.merge_form_results_dls()
dataoutput.print_master_formulation_results_info()
dataoutput.data_size_aggregation_check()
dataoutput.cleaning_master_df_()
dataoutput.pubchem_data()
dataoutput.merge_master_pubchem()
dataoutput.cols_not_unique_check()
dataoutput.column_categ_check_object_type()
dataoutput.output_files()