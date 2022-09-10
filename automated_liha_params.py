import numpy as np
import pandas as pd
import os


class LiHa_Params():

    def __init__(self, file, al_folder):
        self.file = file
        self.al_folder = al_folder
        self.pubchem_data = pd.read_csv(
            r"/Users/calvin/Library/CloudStorage/OneDrive-Personal/Documents/2022/Data_Output/pubchem_data.csv")

    def read_al_file(self):
        file_path = os.path.join(self.al_folder, self.file)
        self.dataframe = pd.read_excel(file_path, skiprows=np.arange(13))
        self.dataframe ['mw_cp_1'] = self.dataframe ['mw_cp_1'].round(2)
        self.dataframe['mw_cp_2']  = self.dataframe ['mw_cp_2'].round(2)



    def reactant_weights(self, lipid_vol, drug_vol, peg_vol):
        self.dataframe['Aqn_Vol'] = self.dataframe['Final_Vol'] * 1000 * (1-self.dataframe['Lipid_Vol_Pcnt'])
        self.dataframe['Lipid_Vol'] = self.dataframe['Final_Vol'] * 1000 * self.dataframe['Lipid_Vol_Pcnt']
        self.dataframe['cmpnd_1_vol'] = self.dataframe['Final_Vol'] * 1000 * self.dataframe['Lipid_Vol_Pcnt'] * \
                                        self.dataframe['Ratio_1']
        self.dataframe['cmpnd_2_vol'] = self.dataframe['Final_Vol'] * 1000 * self.dataframe['Lipid_Vol_Pcnt'] * \
                                        self.dataframe['Ratio_2']
        self.dataframe['cmpnd_4_vol'] = self.dataframe['Final_Vol'] * 1000 * self.dataframe['Lipid_Vol_Pcnt'] * \
                                        self.dataframe['Ratio_4']

        react_names = self.pubchem_data[['compound', 'mw']]
        react_names['mw'] =react_names['mw'].round(2)
        self.dataframe_cmpnd = self.dataframe.merge(react_names, left_on='mw_cp_1', right_on='mw', how='left')
        self.dataframe_cmpnd.drop(columns='mw',inplace=True)
        self.dataframe_cmpnd = self.dataframe_cmpnd.merge(react_names, left_on='mw_cp_2',right_on='mw',how='left')
        self.dataframe_cmpnd.drop(columns='mw',inplace=True)
        self.dataframe_cmpnd['cmpnd_1_weight'] = ((self.dataframe_cmpnd['Concentration_1 (mM)'] * (10 ** (-3))) * (lipid_vol *(10**(-3)))) * self.dataframe_cmpnd['mw_cp_1']
        self.dataframe_cmpnd['cmpnd_2_weight'] = ((self.dataframe_cmpnd['Overall_Concentration_2'] * (10 ** (-3))) * (drug_vol *(10**(-3)))) * self.dataframe_cmpnd['mw_cp_2']
        self.dataframe_cmpnd['cmpnd_4_weight'] = ((self.dataframe_cmpnd['Concentration_4'] * (10 ** (-3))) * (peg_vol *(10**(-3)))) * 2805.5 #MW of Peg
        file_name = os.path.join(self.al_folder,"LiHa_Params.xlsx")
        self.dataframe_cmpnd.to_excel(file_name, index=False)

        print(self.dataframe_cmpnd)

test = LiHa_Params("SVR_RFE_Regressor_CatBoostReg_Ouput_Selection_max_std_sampling_20220902.xlsx",
                   r'/Users/calvin/Library/CloudStorage/OneDrive-Personal/Documents/2022/RegressorCommittee_Output/AL_Random_Output/5NW6N8QY5F_20220902_1119_complete_iteration_12')
test.read_al_file()
test.reactant_weights(0.5, 0.5, 0.3)
