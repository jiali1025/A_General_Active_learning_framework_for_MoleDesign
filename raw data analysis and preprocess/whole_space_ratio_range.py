import pandas as pd
import matplotlib.pyplot as plt

'''
This is the code used to generate the stat of the ratio of the raw data
Which indicates there is imbalance problem
In addition, there may be a need to filter unreasonable data for example very high ratio or negative ratio. This kind of
non-physical data can root from the error in ML-xtb
'''
data_folder_path = '/Users/lijiali/Desktop/Active leanring Framework/data'
figure_save_folder = '/Users/lijiali/Desktop/Active leanring Framework/Figures'


def cal_num_sensitizer(df):
    num = df[(df['T1/S1']>=0.95) & (df['T1/S1']<=1)].count()
    return num

def cal_num_emitter(df):
    num = df[(df['T1/S1']>=0.5) & (df['T1/S1']<=0.53)].count()
    return num
df = pd.read_csv('./preprocessed_data/whole_design_space_ratio.csv',index_col=False)

max = max(df['T1/S1'])
min = min(df['T1/S1'])

print('Max is {max} \n'.format(max=max))
print('Min is {min}'.format(min=min))
df_filter = df[(df['T1/S1']>=0) & (df['T1/S1']<=1.2)]

unfilter_num = df.shape[0]
filter_num = df_filter.shape[0]
print('unfilter row is {unfilter_num} \n'.format(unfilter_num=unfilter_num))
print('filter row is {filter_num}'.format(filter_num=filter_num))
num_sen = cal_num_sensitizer(df_filter)
num_emit = cal_num_emitter(df_filter)
print('num_sensitizer: ' + str(num_sen))
print('num_emitter: ' + str(num_emit))

plt.hist(x=df_filter['T1/S1'],bins=200,
        color="steelblue",
        edgecolor="black")
# plt.ylim((0,1000))
plt.xlim((0,1.2))
plt.xlabel("ratio")
plt.ylabel("frequency")
plt.title("whole_design_space_ratio")
plt.savefig(figure_save_folder + "/whole_design_space_ratio.png")

'''
Generate filtered ratio data
'''
df_filter = df_filter.iloc[:,1:]
df_filter.reset_index(drop=True, inplace=True)

df_filter.to_csv('./preprocessed_data/filtered_whole_design_space_ratio.csv')
