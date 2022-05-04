import pandas as pd
test_df_emit = pd.read_csv('/home/pengfei/projects/AL_framework/Predicted_Results/whole_weighted_emit.csv')
test_df_emit['S1_mae'] = abs(test_df_emit['S1_xTB_ML'] - test_df_emit['S1'])
test_df_emit['T1_mae'] = abs(test_df_emit['T1_xTB_ML'] - test_df_emit['T1'])
test_mae_df_emit = test_df_emit[['SMILES', 'S1_mae','T1_mae']]
avg_sum_s1_mae = test_mae_df_emit['S1_mae'].sum()/test_mae_df_emit.shape[0]
avg_sum_T1_mae = test_mae_df_emit['T1_mae'].sum()/test_mae_df_emit.shape[0]
print('The test S1 mae on emit is {0}'.format(avg_sum_s1_mae))
print('The test T1 mae on emit is {0}'.format(avg_sum_T1_mae))

test_df_sens = pd.read_csv('/home/pengfei/projects/AL_framework/Predicted_Results/whole_weighted_sens.csv')
test_df_sens['S1_mae'] = abs(test_df_sens['S1_xTB_ML'] - test_df_sens['S1'])
test_df_sens['T1_mae'] = abs(test_df_sens['T1_xTB_ML'] - test_df_sens['T1'])
test_mae_df_sens = test_df_sens[['SMILES', 'S1_mae','T1_mae']]
avg_sum_s1_mae = test_mae_df_sens['S1_mae'].sum()/test_mae_df_sens.shape[0]
avg_sum_T1_mae = test_mae_df_sens['T1_mae'].sum()/test_mae_df_sens.shape[0]
print('The test S1 mae on sens is {0}'.format(avg_sum_s1_mae))
print('The test T1 mae on sens is {0}'.format(avg_sum_T1_mae))

test_df_rand = pd.read_csv('/home/pengfei/projects/AL_framework/Predicted_Results/whole_weighted_rand.csv')
test_df_rand['S1_mae'] = abs(test_df_rand['S1_xTB_ML'] - test_df_rand['S1'])
test_df_rand['T1_mae'] = abs(test_df_rand['T1_xTB_ML'] - test_df_rand['T1'])
test_mae_df_rand = test_df_rand[['SMILES', 'S1_mae','T1_mae']]
avg_sum_s1_mae = test_mae_df_rand['S1_mae'].sum()/test_mae_df_rand.shape[0]
avg_sum_T1_mae = test_mae_df_rand['T1_mae'].sum()/test_mae_df_rand.shape[0]
print('The test S1 mae on rand is {0}'.format(avg_sum_s1_mae))
print('The test T1 mae on rand is {0}'.format(avg_sum_T1_mae))