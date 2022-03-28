import pandas as pd

'''
This is the code used to generate the ratio information for the initial training data
'''
data_folder_path = '/Users/lijiali/Desktop/Active leanring Framework/data'

df = pd.read_csv(data_folder_path + '/initial_training/init_train_AL_xTB_ML.csv')

df['T1/S1'] = df['T1_xTB_ML'] / df['S1_xTB_ML']

df.to_csv('./preprocessed_data/init_train_ratio.csv', index=False)

