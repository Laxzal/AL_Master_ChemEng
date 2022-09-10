import os
import re

import numpy as np
import pandas as pd
from scipy.stats import zscore


class FormulationsComplete():

    def __init__(self, folder_stack=None, reg_input_folder=None):
        self.folder_stack = folder_stack
        self.files_formulations = []
        self.reg_input_folder = reg_input_folder
        self.files = []
        self.dls_data = pd.DataFrame()

        self.random_dataframe = pd.DataFrame()
        self.output_dataframe = pd.DataFrame()


        self.columns = ['original_index',
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
                                               'mw_cp_2']
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
        self.combined_data.drop(columns=['Unnamed: 27', '_merge'], inplace=True)
        self.combined_data.reset_index(drop=True)

    def duplicate_index(self):
        self.combined_data.drop_duplicates(subset=self.combined_data.columns.difference(['file_name','location']), inplace=True)
        duplication = self.combined_data[self.combined_data.duplicated(['original_index'], keep=False)]

        print(duplication)

    def only_random_choices(self):

        random_choices = pd.read_excel(r'/Users/calvin/Library/CloudStorage/OneDrive-Personal/Documents/2022/RegressorCommittee_Output/Random_Output/random_choices_approach.xlsx')
        random_choices.rename(columns={'Unnamed: 0': 'original_index'}, inplace=True)

        random_merge = random_choices.merge(self.combined_data['original_index'].to_frame(), right_on='original_index', left_on='original_index', how='inner')
        self.combined_data = random_choices
    def collect_csv(self, date_filter= None):
        '''
         This function will call on the folder that was initalised when the class was called, and 'walk'
        through it to collect all files, place them in a list. Once collected, it will check for only csv/CSV
        files. These csv files will be read in by Pandas (encoding is due to the DLS data being in Hebrew) and that
        data will be appended to the main data frame.
        :return: A dataframe of the DLS data
        '''
        for (dirpath, dirnames, filenames) in os.walk(self.reg_input_folder):
            self.files.extend(filenames)

        for f in self.files:
            if date_filter is None:
                if f.endswith(('.csv', '.CSV')):
                    print(str(f))
                    dw = pd.read_csv(os.path.join(self.reg_input_folder, f), encoding="ISO-8859-8")
                    self.dls_data = pd.concat([self.dls_data, dw], ignore_index=True)
            else:
                if f.endswith(('.csv', '.CSV')):
                    capture = re.findall(r"^(\d+)", f)
                    if int(capture[0])>=int(date_filter):
                        print(str(f))
                        dw = pd.read_csv(os.path.join(self.reg_input_folder, f), encoding="ISO-8859-8")
                        self.dls_data = pd.concat([self.dls_data, dw], ignore_index=True)

        return self.dls_data

    def clean_dls_data(self):
        self.dls_data['Sample Name'] = self.dls_data['Sample Name'].str.strip()  # Clean up white spaces
        self.dls_data['Sample Name'] = self.dls_data['Sample Name'].str.replace("(?=)(\s)(\d)", "",
                                                                                regex=True)  # Remove the numbers proceeding the D(value)
        self.dls_data['Sample Name'] = self.dls_data['Sample Name'].str.replace("(\s)(?=\-[D])", "", regex=True)
        self.dls_data['Sample Name'] = self.dls_data['Sample Name'].str.replace("(?<=\d)(-)(?=[D])", "_", regex=True)
        self.dls_data['Sample Name'] = self.dls_data['Sample Name'].str.replace("(?<=\d)(?=\-)", " ",
                                                                                regex=True)
        self.dls_data['Sample Name'] = self.dls_data['Sample Name'].str.replace("(\s)(?<=)(\-)(?=)(\s)", "_",
                                                                                regex=True)  # Change the - into an _
        self.dls_data['Sample Name'] = self.dls_data['Sample Name'].str.replace("(?<=[A-Z]|\d)(\s)(?=\D)", "_",
                                                                                regex=True)  # Put an underscore between the GUID and D(value)
        self.dls_data['Sample Name'] = self.dls_data['Sample Name'].str.replace("(_FILTERED)", "",
                                                                                regex=True)  # Unique instance of putting the
        self.dls_data['Sample Name'] = self.dls_data['Sample Name'].str.replace("(?<=\d)(-)", "_", regex=True)
        # word filtered in dls naming
        self.dls_data['Sample Name'] = self.dls_data['Sample Name'].str.replace("(_FILTERED\d+)", "", regex=True)
        self.dls_data['Sample Name'] = self.dls_data['Sample Name'].str.strip()  # For good measure

        self.list_dls_unique_scans = list(self.dls_data['Sample Name'].unique())
        print("Number of unique DLS scans:", self.dls_data['Sample Name'].nunique())
        print("DLS Samples Scanned: ", *iter(self.list_dls_unique_scans), sep=' | ')

    def combine_results_formulations(self):
        dls_scan_names = self.dls_data['Sample Name'].str.replace("(?<=_)([A-Z]\d+)", "", regex=True)
        dls_scan_names = dls_scan_names.str.replace("_", "")
        dls_unique_scan_names = dls_scan_names.unique()
        dls_unique_scan_names = pd.DataFrame(dls_unique_scan_names.reshape(-1, 1), columns=['dls_smpl_name'])
        dls_unique_scan_names = dls_unique_scan_names.astype(int)
        # need to remove D parameter
        self.merge = self.combined_data.merge(dls_unique_scan_names, left_on=['original_index'],
                                              right_on=['dls_smpl_name'], indicator=True, how='left')
        self.found_results = self.merge[self.merge['_merge'] == 'both']
        print(self.found_results)

    def selection_columns(self):

        self.formulations = self.found_results[self.columns]
        no_lipid_vol_pcnt = self.formulations.columns.difference(['Lipid_Vol_Pcnt'])
        self.formulations[no_lipid_vol_pcnt] = np.round(self.formulations[no_lipid_vol_pcnt], decimals=2)
        self.formulations_wo_index = self.found_results[self.formulations.columns.difference(['original_index'])]
        print(self.formulations.columns.difference(['original_index']))
    def unlabelled_formulation_load(self):
        self.unlabelled_formulations = pd.read_csv(r"/Users/calvin/Library/CloudStorage/OneDrive-Personal/Documents/2022/RegressorCommittee_Output/AL_Output/KC9T24ZOUI_20220820_0028_complete_iteration_0/Unlabeled_Data.csv")
        self.unlabelled_formulations.drop(columns=['Unnamed: 0'], inplace=True)
        self.unlabelled_formulations = self.unlabelled_formulations.drop_duplicates(subset=self.unlabelled_formulations.columns.difference(['original_index']))
        unlabelled_formulations_no_lipid = self.unlabelled_formulations.columns.difference(['Lipid_Vol_Pcnt'])
        self.unlabelled_formulations[unlabelled_formulations_no_lipid] = np.round(self.unlabelled_formulations[unlabelled_formulations_no_lipid], decimals=2)
    def unlabelled_index_find(self):

        mask = self.combined_data.columns.isin(self.formulations_wo_index.columns)
        df = self.combined_data[self.combined_data.columns[mask]]
        self.formulations_wo_index = self.formulations_wo_index.astype(df.dtypes)

        self.unlabelled_formulations.drop_duplicates(
            subset=self.unlabelled_formulations.columns.difference(['original_index']), inplace=True)
        temp_columns_unlabelled = self.unlabelled_formulations[self.formulations_wo_index.columns]
        merge_dfs = self.unlabelled_formulations.merge(self.formulations, left_on=list(self.formulations_wo_index),
                                             right_on=list(self.formulations_wo_index), how='outer', indicator=True)



        self.found_results = merge_dfs[merge_dfs['_merge'] == 'both']
        self.found_results['original_index_x'] =self.found_results['original_index_x'].astype(int)
        self.found_results['original_index_y'] = self.found_results['original_index_y'].astype(int)
        print(self.found_results)

        self.missing_result = merge_dfs[merge_dfs['_merge'] == 'right_only']

    def filter_out_D_iteration(self):

        # Create the filter list of GUID
        guid_list = list(self.dls_data['Sample Name'].str.replace("(?:\_\D\d)", "", regex=True).unique())
        self.temp_dls_data = pd.DataFrame()

        removal = 0
        for index in guid_list:
            temp_df = self.dls_data[self.dls_data['Sample Name'].str.contains(str(index))]
            scan_iterations = list(temp_df['Sample Name'].str.replace("(\d+_)(?=\D)", "", regex=True).unique())
            sorted_scan_iterations = sorted(scan_iterations, key=lambda x: int("".join([i for i in x if i.isdigit()])),
                                            reverse=True)
            temp_df = temp_df[temp_df['Sample Name'].str.contains(str(sorted_scan_iterations[0]))]
            self.temp_dls_data = pd.concat([self.temp_dls_data, temp_df], ignore_index=True)

            removal += (len(sorted_scan_iterations) - 1)

        print("Number of scans removed: ", removal)
        self.dls_data = self.temp_dls_data
        self.list_dls_unique_scans = list(self.dls_data['Sample Name'].unique())
        print("Number of unique DLS scans[Updated] :", self.dls_data['Sample Name'].nunique())

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
            temp_df = self.dls_data[self.dls_data['Sample Name'] == index]
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
        dlsdata_average['Sample Name'] = dlsdata_average['Sample Name'].str.replace("(?<!\_)(\D+\d)", "",
                                                                                    regex=True)

        dlsdata_average['Sample Name'] = pd.to_numeric(dlsdata_average['Sample Name'])

        self.dlsdata_average = dlsdata_average

    def join_dls_formulations(self):
        self.complete_data = self.found_results.merge(self.dlsdata_average, left_on=['original_index_y'], right_on=['Sample Name'])

    def find_missing_index(self):
        temp_unlabelled_formulations = self.unlabelled_formulations[self.formulations_wo_index.columns]
        self.missing_result = self.missing_result[self.formulations_wo_index.columns]
        for index, row in self.missing_result.iterrows():
            print(index)
            print(row)

            filter_unlabelled_formulations = temp_unlabelled_formulations[temp_unlabelled_formulations['Concentration_1 (mM)']==row[0]]
            filter_unlabelled_formulations = filter_unlabelled_formulations[filter_unlabelled_formulations['Concentration_4'] == row[1]]
            filter_unlabelled_formulations = filter_unlabelled_formulations[filter_unlabelled_formulations['Dispense_Speed_uls']==row[2]]
            filter_unlabelled_formulations = filter_unlabelled_formulations[filter_unlabelled_formulations['Final_Vol']==row[3]]
            filter_unlabelled_formulations = filter_unlabelled_formulations[filter_unlabelled_formulations['Lipid_Vol_Pcnt']==row[4]]
            filter_unlabelled_formulations = filter_unlabelled_formulations[filter_unlabelled_formulations['Overall_Concentration_2']==row[5]]
            filter_unlabelled_formulations = filter_unlabelled_formulations[filter_unlabelled_formulations['Ratio_1']==row[6]]
            filter_unlabelled_formulations = filter_unlabelled_formulations[filter_unlabelled_formulations['Ratio_2']==row[7]]
            filter_unlabelled_formulations = filter_unlabelled_formulations[filter_unlabelled_formulations['Ratio_4']==row[8]]
            filter_unlabelled_formulations = filter_unlabelled_formulations[filter_unlabelled_formulations['mw_cp_1']==row[9]]
            filter_unlabelled_formulations = filter_unlabelled_formulations[filter_unlabelled_formulations['mw_cp_2']==row[10]]

            print (filter_unlabelled_formulations)


   #def pubchem_fillout(self):

    def select_random(self, iterations, instances):
        total = iterations * instances

        random_choices = self.complete_data.sample(n=100, random_state=42)

        random_choices.to_excel(r"/Users/calvin/Library/CloudStorage/OneDrive-Personal/Documents/2022/RegressorCommittee_Output/Random_Output/random_choices.xlsx", index=False)



folder_stack = [#r'/Users/calvin/Library/CloudStorage/OneDrive-Personal/Documents/2022/RegressorCommittee_Output',
                ##r'/Users/calvin/Library/CloudStorage/OneDrive-Personal/Documents/2022/RegressorCommittee_Output/AL_Output_Prev Iteration_EthanolDil_Issue',
                #r'/Users/calvin/Library/CloudStorage/OneDrive-Personal/Documents/2022/RegressorCommittee_Output/AL_Output - Rectified ETHANOLDILIssue',
                #r'/Users/calvin/Library/CloudStorage/OneDrive-Personal/Documents/2022/RegressorCommittee_Output/To be sorted',
                #r'/Users/calvin/Library/CloudStorage/OneDrive-Personal/Documents/2022/RegressorCommittee_Output/random_old',
                #r'/Users/calvin/Library/CloudStorage/OneDrive-Personal/Documents/2022/RegressorCommittee_Output/AL_Output_temp_out',
                #r'/Users/calvin/Library/CloudStorage/OneDrive-Personal/Documents/2022/RegressorCommittee_Output/AL_OUTPUT_Current'
    r'/Users/calvin/Library/CloudStorage/OneDrive-Personal/Documents/2022/RegressorCommittee_Output/AL_Random_Output']

initialisation = FormulationsComplete(folder_stack=folder_stack,
                                      reg_input_folder=r'/Users/calvin/Library/CloudStorage/OneDrive-Personal/Documents/2022/RegressorCommittee_Input')
initialisation.collect_csv(date_filter='20220826')
initialisation.clean_dls_data()
initialisation.regression_output_files()
initialisation.duplicate_index()
initialisation.only_random_choices()
initialisation.combine_results_formulations()
initialisation.selection_columns()
initialisation.unlabelled_formulation_load()
initialisation.unlabelled_index_find()

initialisation.filter_out_D_iteration()
initialisation.z_scoring()
initialisation.join_dls_formulations()
initialisation.select_random(iterations=10, instances=10)