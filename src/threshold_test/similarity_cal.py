import pandas as pd
import rdkit.Chem as Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from tqdm import tqdm

dt_list = []
dt_tot = pd.read_csv('../data_tot/dt_tot_round6.csv')
round = 6
for i in range(16):
    dt_path = f'../cut_data/pred_cut/round_{round}/dt_tot_{i}.csv'
    dt = pd.read_csv(dt_path)
    dt_list.append(dt)
pred_tot = pd.concat(dt_list).dropna()
pred_tot.reset_index(drop = True, inplace = True)
pred_tot = pred_tot[~pred_tot['xTB_S1'].isin(['Invalid SMILES'])]
pred_tot['uncertainty_tot'] = pred_tot['xTB_S1_epi_unc'].apply(lambda x: float(x)) + pred_tot['xTB_T1_epi_unc'].apply(lambda x: float(x))
pred_tot.sort_values(by = 'uncertainty_tot', ascending = False, inplace = True)

suggest_list = []
total_fingerprint = []
pred_len = len(pred_tot)
similar_value_threshold = 0.4
similar_num_threshold = 5

total_smiles = pred_tot['SMILES']
for smile in tqdm(total_smiles):
    mol = Chem.MolFromSmiles(smile)
    fingerprint = AllChem.GetMorganFingerprint(mol, 2)
    total_fingerprint.append(fingerprint)

for i in tqdm(range(pred_len)):
    suggest_list.append(i)
    k = 0
    query_fingerprint = total_fingerprint[i]
    if i >= similar_num_threshold:
        target_fingerprints = [total_fingerprint[j] for j in suggest_list]
        scores = DataStructs.BulkTanimotoSimilarity(query_fingerprint, target_fingerprints)
        total_similar_num = len(list(filter(lambda x: x > similar_value_threshold, scores)))
        if total_similar_num > similar_num_threshold:
            suggest_list.pop()
        if len(suggest_list) >= 20000:
            break
        if (len(suggest_list) % 200) == 0:
            print(f'suggest list size is {len(suggest_list)}')

pred_index = pred_tot.index[suggest_list]
newtrain = dt_tot.iloc[pred_index, :]
train_set = pd.read_csv(f'../train_set/train_round{round}.csv')
train_set = pd.concat([newtrain, train_set])
train_set.to_csv(f'../train_set/train_round{round+1}.csv')

dt_tot_new = dt_tot.iloc[list(set(pred_tot.index)-set(pred_index)), :]
dt_tot_new.reset_index(drop = True, inplace = True)
dt_tot_new.to_csv(f'../data_tot/dt_tot_round{round+1}.csv')