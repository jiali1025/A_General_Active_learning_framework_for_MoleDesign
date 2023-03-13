import pandas as pd
import ast
import collections

df_simple = pd.read_csv('/home/pengfei/projects/AL_framework/Unified_Active_Learning_Framework/Pretrain_chemprop/Pretrain_data/pretrain_demo.csv')
df_complex = pd.read_csv('/home/pengfei/projects/AL_framework/Unified_Active_Learning_Framework/Pretrain_chemprop/Pretrain_data/pretrain_demo_comlex.csv')
simple_label_list = []
complex_label_list = []
for i in range(len(df_simple)):
    simple_label_list.extend(ast.literal_eval(df_simple.iloc[i,3]))
for i in range(len(df_complex)):
    complex_label_list.extend(ast.literal_eval(df_complex.iloc[i,3]))

frequency_simple = collections.Counter(simple_label_list)
frequency_complex = collections.Counter(complex_label_list)
frequency_simple = dict(frequency_simple)
sum_simple = sum(frequency_simple.values())
frequency_complex = dict(frequency_complex)
sum_complex = sum(frequency_complex.values())
percent_frequency_simple = {key: value / sum_simple for key, value in frequency_simple.items()}
percent_frequency_complex = {key: value / sum_complex for key, value in frequency_complex.items()}
print(sorted(percent_frequency_simple.values(), reverse=True))
print(sorted(percent_frequency_complex.values(), reverse=True))

print('cool')