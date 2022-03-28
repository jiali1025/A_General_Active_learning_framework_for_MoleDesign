import pandas as pd
import matplotlib.pyplot as plt

'''
Used to filter the init_train data
'''

df = pd.read_csv('./preprocessed_data/init_train_ratio.csv')

max = max(df['T1/S1'])
min = min(df['T1/S1'])

print('Max is {max} \n'.format(max=max))
print('Min is {min}'.format(min=min))
df_filter = df[(df['T1/S1']>=0) & (df['T1/S1']<=1.2)]

unfilter_num = df.shape[0]
filter_num = df_filter.shape[0]
print('unfilter row is {unfilter_num} \n'.format(unfilter_num=unfilter_num))
print('filter row is {filter_num}'.format(filter_num=filter_num))

'''
Generate filtered ratio data
'''
df_filter.to_csv('./preprocessed_data/filtered_init_train.csv',index=False)
