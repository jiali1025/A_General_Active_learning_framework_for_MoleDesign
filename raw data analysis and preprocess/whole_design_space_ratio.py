'''
Create the ratio form csv for the training data
'''
import pandas as pd

'''
global variable to store the path of the target folders change it to your local path
'''
data_folder_path = '/Users/lijiali/Desktop/Active leanring Framework/data'
figure_save_folder = '/Users/lijiali/Desktop/Active leanring Framework/Figures'

df = pd.read_csv(data_folder_path + '/whole_design_space/xtb_ML_calcs_compiled_noTest.csv', index_col=False)

df['T1/S1'] = df['T1_xTB_ML'] / df['S1_xTB_ML']

df.to_csv('./preprocessed_data/whole_design_space_ratio.csv', index=False)


