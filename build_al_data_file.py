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
        self.dlsdata_average = dlsdata_average

    def collect_formulations(self):
        for (dirpath, dirnames, filenames) in os.walk(self.folder_formulation):
            dirnames[:] = [d for d in dirnames if '_complete' in d]
            for file in filenames:
                if file.endswith(('.xlsx', '.XLSX')):
                    self.files_formulations.append(os.path.join(dirpath, file))


        for f in self.files_formulations:
            if f.endswith(('.xlsx', '.XLSX')):
                if 'random' in f:
                    print(str(f))
                    tdf_random = pd.read_excel(f)
                    self.random_dataframe = pd.concat([self.random_dataframe, tdf_random], ignore_index=True)
                    # self.dls_data = self.dls_data.append(dw, ignore_index=True)
                elif 'Output' in f:
                    # Need to hardcode the skiprows for Output
                    tdf_output = pd.read_excel(f, skiprows=np.arange(13))
                    self.output_dataframe = pd.concat([self.output_dataframe, tdf_output], ignore_index=True)
                    # header=0)
                    print(tdf_output)

    def merge_dls_data(self):

        results_comp_df = pd.merge(self.output_dataframe, self.dlsdata_average,
                                   how="outer",
                                   left_on="original_index",
                                   right_on="Sample Name").reset_index(drop=True)





##Test
test = ALDataBuild(
    folder_dls=r'/Users/calvin/Library/CloudStorage/OneDrive-Personal/Documents/2022/RegressorCommittee_Input',
    folder_formulations=r'/Users/calvin/Library/CloudStorage/OneDrive-Personal/Documents/2022/RegressorCommittee_Output')

test.collect_csv()
test.clean_dls_data()
test.filter_out_D_iteration()
test.z_scoring(threshold=1.0)
test.collect_formulations()
test.merge_dls_data()
