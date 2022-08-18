import pandas as pd
import os
import numpy as np


class Dls_Raw_Results():

    def __init__(self, folder=None, folder_stack=None, column_selection=None):
        self.folder = folder
        self.files = []
        self.dls_data = None
        self.found = None
        self.folder_stack = folder_stack
        self.files_formulations = []
        self.column_selection = column_selection
        self.random_dataframe = pd.DataFrame()
        self.output_dataframe = pd.DataFrame()

    def collect_csv(self):
        '''
         This function will call on the folder that was initalised when the class was called, and 'walk'
        through it to collect all files, place them in a list. Once collected, it will check for only csv/CSV
        files. These csv files will be read in by Pandas (encoding is due to the DLS data being in Hebrew) and that
        data will be appended to the main data frame.
        :return: A dataframe of the DLS data
        '''
        for (dirpath, dirnames, filenames) in os.walk(self.folder):
            self.files.extend(filenames)

        for f in self.files:

            if f.endswith(('.csv', '.CSV')):
                print(str(f))
                dw = pd.read_csv(os.path.join(self.folder, f), encoding="ISO-8859-8")
                self.dls_data = self.dls_data.append(dw, ignore_index=True)

        return self.dls_data

    def clean_dls_data(self):
        self.dls_data['Sample Name'] = self.dls_data['Sample Name'].str.strip()  # Clean up white spaces
        self.dls_data['Sample Name'] = self.dls_data['Sample Name'].str.replace("(?=)(\s)(\d)", "",
                                                                                regex=True)  # Remove the numbers proceeding the D(value)

        self.dls_data['Sample Name'] = self.dls_data['Sample Name'].str.replace("(?<=\d)(?=\-)", " ",
                                                                                regex=True)
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

    def regression_output_files(self):
        for folder_formulation in self.folder_stack:
            for (dirpath, dirnames, filenames) in os.walk(folder_formulation):
                dirnames[:] = [d for d in dirnames if '_complete' in d]
                for file in filenames:
                    if file.endswith(('.xlsx', '.XLSX')):
                        self.files_formulations.append(os.path.join(dirpath, file))
            ###Need to remove 'cv_results' as these are large files and grinds the for loop to a halt
            self.files_formulations[:] = [d for d in self.files_formulations if 'cv_results' not in d]
            self.files_formulations[:] = [d for d in self.files_formulations if 'LiHa_Params' not in d]
            for f in self.files_formulations:
                if f.endswith(('.xlsx', '.XLSX')):

                    print(f)
                    if 'random_unlabeled' in f:
                        # print(str(f))
                        tdf_random = pd.read_excel(f)
                        tdf_random['location'] = 'random'
                        tdf_random['file_name'] = str(f)
                        self.random_dataframe = pd.concat([self.random_dataframe, tdf_random], ignore_index=True)

                        # self.dls_data = self.dls_data.append(dw, ignore_index=True)
                    elif 'Ouput_Selection_max_std_sampling' in f:
                        # Need to hardcode the skiprows for Output
                        tdf_output = pd.read_excel(f, skiprows=np.arange(13))
                        tdf_output['location'] = 'al'
                        tdf_output['file_name'] = str(f)
                        self.output_dataframe = pd.concat([self.output_dataframe, tdf_output], ignore_index=True)
                        # header=0)
                        # print(tdf_output)
        self.random_dataframe.drop(columns=['original_index'], inplace=True)
        self.random_dataframe.rename(columns={'Unnamed: 0': 'original_index'}, inplace=True)
        self.output_dataframe.drop(columns=['sample_scoring'], inplace=True)

        self.combined_data = pd.concat([self.output_dataframe, self.random_dataframe])
        self.combined_data['original_index'] = self.combined_data['original_index'].astype(int)
        self.combined_data.reset_index(drop=True)

        self.combined_data = self.combined_data[self.column_selection]

    def search(self, formulation_id):
        if self.dls_data is not None:
            self.found = self.dls_data[self.dls_data['Sample Name'].str.contains(formulation_id)]
        else:
            self.found = self.combined_data[self.combined_data['original_index'] == int(formulation_id)]
        print(self.found)
        print(self.found.index)

    def search_via_formulation(self, formulation_data):

        formulation_data = pd.read_excel(formulation_data)
        # Compare column names first
        #formulation_data_selection = formulation_data[self.column_selection]
        formulation_column_selection = list(formulation_data.columns.intersection(self.combined_data.columns))
        temp_columns_combined_data = list(self.combined_data.columns)
        temp_columns_combined_data.remove('original_index')
        temp_columns_combined_data.remove('location')
        temp_columns_combined_data.remove('file_name')
        formulation_data.loc[formulation_data['mw_cp_2'] == 430.6999999999999, 'mw_cp_2'] = 430.7
        formulation_data.loc[formulation_data['Ratio_2'] == 0.44999999999999996, 'Ratio_2'] = 0.45
        formulation_data.loc[formulation_data['Ratio_2'] == 0.4499999999999999, 'Ratio_2'] = 0.45

        formulation_data = formulation_data[formulation_column_selection]



        mask = self.combined_data.columns.isin(formulation_data.columns)
        df = self.combined_data[self.combined_data.columns[mask]]
        formulation_data = formulation_data.astype(df.dtypes)

        print('Combined Dataframe Column Info')
        print(self.combined_data.info())
        print('Formulation Dataframe Column Info')
        print(formulation_data.info())

        merge_dfs = self.combined_data.merge(formulation_data.drop_duplicates(), left_on=temp_columns_combined_data,
                                             right_on=list(formulation_data),
                                             how='left', indicator=True)
        found_results = merge_dfs[merge_dfs['_merge'] == 'both']
        found_results.drop_duplicates(inplace=True)
        file_name = os.path.join(r"/Users/calvin/Library/CloudStorage/OneDrive-Personal/Documents/2022/RegressorCommittee_Output/","joined_files.xlsx")
        print(found_results)

        found_results.to_excel(file_name)
##Test

folder_stack = [r'/Users/calvin/Library/CloudStorage/OneDrive-Personal/Documents/2022/RegressorCommittee_Output',
                r'/Users/calvin/Library/CloudStorage/OneDrive-Personal/Documents/2022/RegressorCommittee_Output/AL_Output_Prev Iteration_EthanolDil_Issue',
                r'/Users/calvin/Library/CloudStorage/OneDrive-Personal/Documents/2022/RegressorCommittee_Output/AL_Output - Rectified ETHANOLDILIssue',
                r'/Users/calvin/Library/CloudStorage/OneDrive-Personal/Documents/2022/RegressorCommittee_Output/To be sorted',
                r'/Users/calvin/Library/CloudStorage/OneDrive-Personal/Documents/2022/RegressorCommittee_Output/Random_Output',
                r'/Users/calvin/Library/CloudStorage/OneDrive-Personal/Documents/2022/RegressorCommittee_Output/AL_Output_temp_out',
                r'/Users/calvin/Library/CloudStorage/OneDrive-Personal/Documents/2022/RegressorCommittee_Output/AL_OUTPUT_Current']

test = Dls_Raw_Results(folder_stack=folder_stack, column_selection=['original_index',
                                                                    'Concentration_1 (mM)',
                                                                    'Ratio_1',
                                                                    'Overall_Concentration_2',
                                                                    'Ratio_2',
                                                                    'Concentration_4',
                                                                    'Ratio_4',
                                                                    'Final_Vol',
                                                                    'Lipid_Vol_Pcnt',
                                                                    'Dispense_Speed_uls',
                                                                    'mw_cp_1',
                                                                    #'xlogp_cp_1',
                                                                    #'complexity_cp_1',
                                                                    #'heavy_atom_count_cp_1',
                                                                    #'single_bond_cp_1',
                                                                    #'double_bond_cp_1',
                                                                    'mw_cp_2',
                                                                    #'h_bond_acceptor_count_cp_2',
                                                                    #'xlogp_cp_2',
                                                                    #'complexity_cp_2',
                                                                    #'heavy_atom_count_cp_2',
                                                                    #'tpsa_cp_2',
                                                                    #'ssr_cp_2',
                                                                    #'single_bond_cp_2',
                                                                    #'double_bond_cp_2',
                                                                    #'aromatic_bond_cp_2',
    'location',
                                                                    'file_name'
])
test.regression_output_files()
#test.search(432852)

test.search_via_formulation(
    formulation_data=r'/Users/calvin/Library/CloudStorage/OneDrive-Personal/Documents/2022/RegressorCommittee_Output/formulation_find/formulation_find.xlsx')
