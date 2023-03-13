import pandas as pd

df = pd.read_csv('/home/lijiali1025/projects/AL_framework/Unified_Active_Learning_Framework/Pretrain_chemprop/Pretrain_data/final_pretrain_complex_vocab_ordered.csv')
# df_filered = pd.read_csv('/home/pengfei/projects/AL_framework/Unified_Active_Learning_Framework/Pretrain_chemprop/Pretrain_data/final_pretrain_complex_vocab_filtered_sampled_0.02M.csv')
# df_filered_sampled = df_filered.sample(frac=0.001, replace=False, random_state=1)
# df_filered_sampled.to_csv('/home/pengfei/projects/AL_framework/Unified_Active_Learning_Framework/Pretrain_chemprop/Pretrain_data/final_pretrain_complex_vocab_filtered_sampled_0.2M.csv',index=False)
df_filtered = df[~df['Label'].str.contains('None')]
df_filtered_sampled = df_filtered.sample(frac=0.001, replace=False, random_state=1)
df_filtered_sampled.to_csv('/home/lijiali1025/projects/AL_framework/Unified_Active_Learning_Framework/Pretrain_chemprop/Pretrain_data/final_pretrain_complex_vocab_ordered_0.02M.csv')
# df1.to_csv('/home/pengfei/projects/AL_framework/Unified_Active_Learning_Framework/Pretrain_chemprop/Pretrain_data/final_pretrain_complex_vocab_filtered.csv',index=False)

print('cool')