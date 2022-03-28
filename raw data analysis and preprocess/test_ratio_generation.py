import pandas as pd
'''
Add in the ratio information to test data.
'''
data_folder_path = '/Users/lijiali/Desktop/Active leanring Framework/data'

df_emit = pd.read_csv(data_folder_path + '/test_data/3k_emit_test.csv')
df_rand = pd.read_csv(data_folder_path + '/test_data/3k_rand_test.csv')
df_sens = pd.read_csv(data_folder_path + '/test_data/3k_sens_test.csv')

df_emit['T1/S1'] = df_emit['T1'] / df_emit['S1']
df_rand['T1/S1'] = df_rand['T1'] / df_rand['S1']
df_sens['T1/S1'] = df_sens['T1'] / df_sens['S1']

df_emit.to_csv('./preprocessed_data/3k_emit_test_ratio.csv')
df_rand.to_csv('./preprocessed_data/3k_rand_test_ratio.csv')
df_sens.to_csv('./preprocessed_data/3k_sens_test_ratio.csv')


