import os

import numpy as np
import pandas as pd
from scipy.stats import zscore


class ALDataBuild:

    def __init__(self, folder_dls, folder_formulations):
        self.folder = folder_dls
        self.files = []
        self.dls_data = pd.DataFrame()

        self.folder_formulation = folder_formulations
        self.files_formulations = []

        self.random_dataframe = pd.DataFrame()
        self.output_dataframe = pd.DataFrame()

    def load_prev_gguid(self,save_path, recent_folder):
        prev_gguid_path = os.path.join(save_path,recent_folder,'iteration_prev_gguid.csv')
        self.prev_gguid_df = pd.read_csv(prev_gguid_path)
        self.recent_guid = list(self.prev_gguid_df['gguid'])[-1]

        return self.prev_gguid_df

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
                if str(self.recent_guid) in f:
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

    def collect_formulations(self):
        for (dirpath, dirnames, filenames) in os.walk(self.folder_formulation):
            dirnames[:] = [d for d in dirnames if '_complete' in d]
            for file in filenames:
                if file.endswith(('.xlsx', '.XLSX')):
                    self.files_formulations.append(os.path.join(dirpath, file))
        ###Need to remove 'cv_results' as these are large files and grinds the for loop to a halt
        self.files_formulations[:] = [d for d in self.files_formulations if 'cv_results' not in d]
        for f in self.files_formulations:
            if f.endswith(('.xlsx', '.XLSX')):

                print(f)
                if 'random' in f:
                    # print(str(f))
                    tdf_random = pd.read_excel(f)
                    self.random_dataframe = pd.concat([self.random_dataframe, tdf_random], ignore_index=True)
                    # self.dls_data = self.dls_data.append(dw, ignore_index=True)
                elif 'Output' in f:
                    # Need to hardcode the skiprows for Output
                    tdf_output = pd.read_excel(f, skiprows=np.arange(13))
                    self.output_dataframe = pd.concat([self.output_dataframe, tdf_output], ignore_index=True)
                    # header=0)
                    # print(tdf_output)

        self.output_dataframe = self.output_dataframe[['original_index',
                                                       'Concentration_1 (mM)',
                                                       'Ratio_1',
                                                       'Overall_Concentration_2',
                                                       'Ratio_2',
                                                       'Overall_Concentration_3',
                                                       'Ratio_3',
                                                       'Concentration_4',
                                                       'Ratio_4',
                                                       'Final_Vol',
                                                       'Lipid_Vol_Pcnt',
                                                       'Dispense_Speed_uls',
                                                       'component_1_vol_stock',
                                                       'component_2_vol_conc',
                                                       'component_2_vol_stock',
                                                       'component_3_vol_conc',
                                                       'component_3_vol_stock',
                                                       'ethanol_dil',
                                                       'mw_cp_1',
                                                       'heavy_atom_count_cp_1',
                                                       'single_bond_cp_1',
                                                       'double_bond_cp_1',
                                                       'mw_cp_2',
                                                       'h_bond_donor_count_cp_2',
                                                       'h_bond_acceptor_count_cp_2',
                                                       'heavy_atom_count_cp_2',
                                                       'ssr_cp_2',
                                                       'single_bond_cp_2',
                                                       'double_bond_cp_2',
                                                       'mw_cp_3',
                                                       'h_bond_donor_count_cp_3',
                                                       'h_bond_acceptor_count_cp_3',
                                                       'heavy_atom_count_cp_3',
                                                       'ssr_cp_3',
                                                       'single_bond_cp_3',
                                                       'double_bond_cp_3',
                                                       'sample_scoring']]

        self.output_dataframe.dropna(inplace=True)
        self.random_dataframe.dropna(how='all', inplace=True)
        self.random_dataframe.drop(columns=['original_index'], inplace=True)

        # Random is missing this column name for some reason
        self.random_dataframe.rename(columns={'Unnamed: 0': 'original_index'}, inplace=True)

    def merge_dls_data(self):
        # self.dlsdata_average['Sample Name'] = self.dlsdata_average['Sample Name'].str.strip()
        # self.output_dataframe['original_index'] = self.output_dataframe['original_index'].str.strip()

        self.merged_dataframe_AL = pd.merge(self.output_dataframe, self.dlsdata_average,
                                            how="inner",
                                            left_on="original_index",
                                            right_on="Sample Name").reset_index(drop=True)

        print(self.merged_dataframe_AL)

        self.merged_dataframe_random = pd.merge(self.random_dataframe, self.dlsdata_average,
                                                how="inner",
                                                left_on="original_index",
                                                right_on="Sample Name").reset_index(drop=True)

        ###Don't know why this is happpening but quick fix
        self.merged_dataframe_AL['ethanol_dil'] = 0.00
        self.merged_dataframe_random['ethanol_dil'] = 0.00
        print(self.merged_dataframe_random)

    def load_x_y_prev_run(self, save_path, recent_folder):
        recent_iteration_run = recent_folder
        path_of_file = os.path.join(save_path, recent_iteration_run, "Added_Data.csv")
        if os.path.exists(path_of_file):
            prev_x_y_df = pd.read_csv(path_of_file)
            prev_x_df = prev_x_y_df.drop(columns=['Z-Average (d.nm)']).to_numpy()
            prev_y_df = prev_x_y_df['Z-Average (d.nm)'].to_numpy().reshape(-1)

        else:
            prev_x_df = None
            prev_y_df = None

        return prev_x_df, prev_y_df

    def load_x_val_prev_run(self, save_path, recent_folder):
        # Need most recent folder name
        # Then load the X_val file if exists.
        x_val_prev_index = None
        recent_iteration_run = recent_folder
        path_of_file = os.path.join(save_path, recent_iteration_run, "Unlabeled_Data.csv")
        if os.path.exists(path_of_file):
            X_val_df = pd.read_csv(path_of_file)
            if 'original_index' in X_val_df.columns:
                X_val_df.drop(columns=['original_index'], inplace=True)
            if 'Unnamed: 0' in X_val_df:
                x_val_prev_index = X_val_df['Unnamed: 0'].copy()
                X_val_df.drop(columns=['Unnamed: 0'], inplace=True)
            X_val = X_val_df.to_numpy()
            X_val_column_names = X_val_df.columns

            return X_val, X_val_column_names, x_val_prev_index

    def remove_AL_from_unlabelled(self, X_val, X_val_columns_names,x_prev_index, count):
        X_val = pd.DataFrame(X_val, columns=X_val_columns_names)

        if count<=1:
            temporary_AL_dataframe = self.merged_dataframe_AL.drop(columns=['original_index', 'sample_scoring',
                                                                            'Sample Name', 'Z-Average (d.nm)',
                                                                            'PdI', 'PdI Width (d.nm)'])

            merge_dfs = X_val.merge(temporary_AL_dataframe.drop_duplicates(), on=list(temporary_AL_dataframe),
                                    how='left', indicator=True)
            merge_dfs = merge_dfs[merge_dfs['_merge'] == 'left_only']
            merge_dfs.drop(columns=['_merge'], inplace=True)
            print("No. of removed unlabelled formulations: ", X_val.shape[0] - merge_dfs.shape[0])
            X_val = merge_dfs.to_numpy()

        else:
            temporary_AL_dataframe = self.merged_dataframe_AL.drop(columns=['sample_scoring',
                                                                            'Sample Name', 'Z-Average (d.nm)',
                                                                            'PdI', 'PdI Width (d.nm)'])

            X_val['original_index'] = x_prev_index
            merge_dfs = X_val.merge(temporary_AL_dataframe['original_index'], on=['original_index'], how='left',
                                    indicator=True)
            merge_dfs = merge_dfs[merge_dfs['_merge'] == 'left_only']
            merge_dfs.drop(columns=['_merge','original_index'], inplace=True)
            print("No. of removed unlabelled formulations: ", X_val.shape[0] - merge_dfs.shape[0])
            X_val = merge_dfs.to_numpy()

        return X_val, X_val_columns_names

    def return_AL_data(self):
        pre_x_train = self.merged_dataframe_AL.drop(columns=['original_index', 'sample_scoring',
                                                             'Sample Name', 'Z-Average (d.nm)',
                                                             'PdI', 'PdI Width (d.nm)'])

        self.X_train_AL = pre_x_train.to_numpy()

        pre_y_train = self.merged_dataframe_AL[['Z-Average (d.nm)']]

        self.y_train_AL = pre_y_train.to_numpy().reshape(-1)

        return self.X_train_AL, self.y_train_AL

    def add_AL_to_train(self, X_train_initial, y_train_initial,
                        X_train_prev, y_train_prev):

        if X_train_prev is not None and y_train_prev is not None:
            X_train = np.vstack((X_train_initial, X_train_prev, self.X_train_AL))
            y_train = np.hstack((y_train_initial, y_train_prev, self.y_train_AL)).astype(float)
        else:
            X_train = np.vstack((X_train_initial, self.X_train_AL))
            y_train = np.hstack((y_train_initial, self.y_train_AL)).astype(float)

        return X_train, y_train

    def remove_random_from_unlabelled(self, X_val, X_val_columns_names):
        X_val = pd.DataFrame(X_val, columns=X_val_columns_names)
        if 'sample_scoring' in self.merged_dataframe_random.columns:
            temporary_random_dataframe = self.merged_dataframe_random.drop(columns=['original_index', 'sample_scoring',
                                                                                    'Sample Name', 'Z-Average (d.nm)',
                                                                                    'PdI', 'PdI Width (d.nm)'])
        else:
            temporary_random_dataframe = self.merged_dataframe_random.drop(columns=['original_index',
                                                                                    'Sample Name', 'Z-Average (d.nm)',
                                                                                    'PdI', 'PdI Width (d.nm)'])

        merge_dfs = X_val.merge(temporary_random_dataframe.drop_duplicates(), on=list(temporary_random_dataframe),
                                how='left', indicator=True)
        merge_dfs = merge_dfs[merge_dfs['_merge'] == 'left_only']
        merge_dfs.drop(columns=['_merge'], inplace=True)
        X_val = merge_dfs.to_numpy()

        return X_val, X_val_columns_names

    def return_random_data(self):
        if 'sample_scoring' in self.merged_dataframe_random.columns:
            pre_x_train = self.merged_dataframe_random.drop(columns=['original_index', 'sample_scoring',
                                                                     'Sample Name', 'Z-Average (d.nm)',
                                                                     'PdI', 'PdI Width (d.nm)'])
        else:
            pre_x_train = self.merged_dataframe_random.drop(columns=['original_index',
                                                                     'Sample Name', 'Z-Average (d.nm)',
                                                                     'PdI', 'PdI Width (d.nm)'])

        self.X_train_random = pre_x_train.to_numpy()

        pre_y_train = self.merged_dataframe_random[['Z-Average (d.nm)']]

        self.y_train_random = pre_y_train.to_numpy().reshape(-1)

        return self.X_train_random, self.y_train_random

    def add_random_to_train(self, X_train_initial, y_train_initial,
                            X_train_prev, y_train_prev):

        if X_train_prev is not None and y_train_prev is not None:
            X_train = np.vstack((X_train_initial, X_train_prev, self.X_train_random))
            y_train = np.hstack((y_train_initial, y_train_prev, self.y_train_random)).astype(float)
        else:
            X_train = np.vstack((X_train_initial, self.X_train_random))
            y_train = np.hstack((y_train_initial, self.y_train_random)).astype(float)

        return X_train, y_train

##Test


# X_Val Function
# def unlabelled_data(file, method, column_removal_experiment: list=None):
#     ul_df = pd.read_csv(file)
#     column_drop = ['Duplicate_Check',
#                    'PdI Width (d.nm)',
#                    'PdI',
#                    'Z-Average (d.nm)',
#                    'ES_Aggregation']
#
#
#     ul_df = ul_df.drop(columns=column_drop)
#     #ul_df = ul_df.drop(columns=useless_clm_drop)
#
#     #Remove a column(s)
#     if column_removal_experiment is not None:
#         ul_df.drop(columns=column_removal_experiment, inplace=True)
#
#
#     ul_df.replace(np.nan, 'None', inplace=True)
#     if "Component_1" and "Component_2" and "Component_3" in ul_df.columns:
#         ul_df = pd.get_dummies(ul_df, columns=["Component_1", "Component_2", "Component_3"],
#                                prefix="", prefix_sep="")
#         # if method=='fillna':    ul_df['Component_3'] = ul_df['Component_3'].apply(lambda x: None if pd.isnull(x) else x) #TODO This should be transformed into an IF function, thus when the function for unlabelled is filled with a parameter, then activates
#
#         ul_df = ul_df.groupby(level=0, axis=1, sort=False).sum()
#
#     # print(ul_df.isna().any())
#     X_val = ul_df.to_numpy()
#     columns_x_val = ul_df.columns
#     return X_val, columns_x_val
#
#
# column_removal_experiment = ['xlogp_cp_1', 'xlogp_cp_2', 'xlogp_cp_3',
#                              'aromatic_bond_cp_2', 'complexity_cp_3',
#                              'complexity_cp_2', 'complexity_cp_1', 'Final_Concentration',
#                              'final_lipid_volume', 'component_1_vol', 'component_2_vol',
#                              'component_3_vol', 'component_4_vol', 'component_1_vol_conc',
#                              'tpsa_cp_3', 'tpsa_cp_2',
#                              # 'heavy_atom_count_cp_1','heavy_atom_count_cp_2','heavy_atom_count_cp_3',
#                              # 'single_bond_cp_1','double_bond_cp_1',
#                              # 'single_bond_cp_2','double_bond_cp_2',
#                              # 'single_bond_cp_3','double_bond_cp_3',
#                              # 'h_bond_donor_count_cp_2','h_bond_acceptor_count_cp_2',
#                              # 'h_bond_donor_count_cp_3','h_bond_acceptor_count_cp_3',
#                              # 'ssr_cp_2','ssr_cp_3',
#                              'Req_Weight_1', 'Ethanol_1',
#                              'Req_Weight_2', 'Ethanol_2',
#                              'Req_Weight_3',
#                              'Req_Weight_4', 'Ethanol_4'
#                              # 'ethanol_dil',
#                              # 'component_1_vol_stock',
#                              # 'component_2_vol_stock',
#                              # 'component_3_vol_stock'
#
#                              ]
# X_val, columns_x_val = unlabelled_data(file = r"/Users/calvin/Documents/OneDrive/Documents/2022/Data_Output/unlabelled_data_full.csv",
#                                                  method='fillna',
#                                                  column_removal_experiment=column_removal_experiment)  # TODO need to rename
#
# test = ALDataBuild(
#     folder_dls=r'/Users/calvin/Library/CloudStorage/OneDrive-Personal/Documents/2022/RegressorCommittee_Input',
#     folder_formulations=r'/Users/calvin/Library/CloudStorage/OneDrive-Personal/Documents/2022/RegressorCommittee_Output')
#
# test.collect_csv()
# test.clean_dls_data()
# test.filter_out_D_iteration()
# test.z_scoring(threshold=1.0)
# test.collect_formulations()
# test.merge_dls_data()
# test.remove_from_unlabelled(X_val = X_val, X_val_columns_names=columns_x_val)
