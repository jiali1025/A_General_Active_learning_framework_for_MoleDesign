import pandas as pd

df1 = pd.read_csv('/home/lijiali1025/projects/AL_framework/Unified_Active_Learning_Framework/Pretrain_chemprop/Pretrain_data/CID-SMILES-ATOMS-10M-LARGEMOL.csv')
df2 = pd.read_csv('/home/lijiali1025/projects/AL_framework/Unified_Active_Learning_Framework/Pretrain_chemprop/Pretrain_data/CID-SMILES-ATOMS-10M-SMALLMOL.csv')

df1 = df1['SMILES']
df2 = df2['SMILES']

df3 = pd.concat([df1,df2])
df3.to_csv('./final_pretrain.csv')
