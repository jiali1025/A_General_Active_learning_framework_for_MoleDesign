import pandas as pd
import numpy as np
import os
import rdkit.Chem as Chem
from tqdm import tqdm


'''
First pre-process the data.
Then divide the test set.
Then generate the initial train set and remaining data.
'''

filePath = '../xtb_ml_data' # Extract all data, as well as sensitizer and emitter
xtb_data = os.listdir(filePath)
dt_list = []

j = 0
for i in xtb_data:
    path = filePath + "/" + i
    dt = pd.read_csv(path).iloc[:, :3]
    dt.columns = ['SMILES', 'xTB_S1', 'xTB_T1']
    dt = dt[~dt['xTB_S1'].isin(['Invalid SMILES'])]
    dt['T1S1ratio'] = pd.to_numeric(dt['xTB_T1']) / pd.to_numeric(dt['xTB_S1'])
    dt_list.append(dt)

# Merge all data
dt_tot = pd.concat(dt_list)  
dt_tot = dt_tot.drop_duplicates(subset = 'SMILES', keep=False)
dt_tot.dropna(inplace = True)
dt_tot.reset_index(drop = True, inplace = True)

# Remove all charged molecules
index = []
for i in tqdm(range(len(dt_tot))):
    mol = Chem.MolFromSmiles(dt_tot.iloc[i, 0])
    if mol == None: 
        continue
    Chem.Kekulize(mol)
    if abs(Chem.GetFormalCharge(mol)) == 0:
        index.append(i)

# Screen out sensitizers and emitters
dt_tot = dt_tot.iloc[index, :]
dt_tot.reset_index(drop = True, inplace = True)
dt_emit = dt_tot[(dt_tot['T1S1ratio'] > (1/2.2)) & (dt_tot['T1S1ratio'] < (1/1.8))]
dt_sens = dt_tot[(dt_tot['T1S1ratio'] > 0.8) & (dt_tot['T1S1ratio'] < 1)]

# Divide test sets
np.random.seed(2022)
def is_large(smi): # Split the target data according to atom_num > 20
    mol = Chem.MolFromSmiles(smi)
    atoms_num = mol.GetNumAtoms()
    if atoms_num > 20:
        return True
    if atoms_num <= 20:
        return False

test_rand = dt_tot.sample(n = 3000, replace = False)
test_emit = dt_emit.sample(n = 3000, replace = False)
test_sens = dt_sens.sample(n = 3000, replace = False)
test_tot = pd.concat([test_rand, test_emit, test_sens])  
test_tot = test_tot.drop_duplicates(subset = 'SMILES', keep=False)
test_tot['is_large'] = test_tot['SMILES'].apply(lambda x: is_large(x))
test_large = test_tot[test_tot['is_large'] == True]
test_small = test_tot[test_tot['is_large'] == False]

test_rand.to_csv('../test_set/test_rand.csv')
test_emit.to_csv('../test_set/test_emit.csv')
test_sens.to_csv('../test_set/test_sens.csv')
test_large.to_csv('../test_set/test_large.csv')
test_small.to_csv('../test_set/test_small.csv')
test_tot.to_csv('../test_set/test_tot.csv')

# Generate the initial train set
dt_tot_dup = pd.concat([test_tot.iloc[:, :-1], dt_tot])
dt_tot = dt_tot_dup.drop_duplicates(subset = 'SMILES', keep=False)
dt_tot.reset_index(drop = True, inplace = True)
dt_train = dt_tot.sample(n = 20000, replace = False)
dt_train.to_csv('../train_set/train_round1.csv') 

# Generate the total data remain
dt_tot_round1 = pd.concat([dt_train, dt_tot])
dt_tot_round1 = dt_tot_round1.drop_duplicates(subset = 'SMILES', keep = False)
dt_tot_round1.reset_index(drop = True, inplace = True)
dt_tot_round1.to_csv('../data_tot/dt_tot_round1.csv')