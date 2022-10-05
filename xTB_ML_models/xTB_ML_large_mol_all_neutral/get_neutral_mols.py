import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit.Chem import Descriptors, Draw
from rdkit import Chem
import os
from openbabel import openbabel as ob
from openbabel import pybel
from tqdm import tqdm

df_300k = pd.read_csv('../combined_train_data_300k.csv')
df_largemol = pd.read_csv('combined_data_large_mol_all.csv')
df = df_largemol.merge(df_300k, how='outer')
df.drop_duplicates(subset = 'SMILES', inplace=True)
df.dropna(inplace=True)

charges = []
neutral_smiles = []
for row in tqdm(df.iterrows(), total = len(df)):
    smiles = row[1]['SMILES']
    mol2 = pybel.readstring('smi', smiles)
    charge = mol2.charge
    # print(smiles, charge, deviation)
    charges.append(charge)
    if charge == 0:
        neutral_smiles.append(smiles)

df_neutral = pd.DataFrame(neutral_smiles, columns=['SMILES'])
df_neutral = df_neutral.merge(df, how='inner')
df_neutral.to_csv('combined_data_train_neutral.csv', index=False)

# df_test = pd.read_csv('combined_data_large_mol_test.csv')
# df_test.drop_duplicates(subset = 'SMILES', inplace=True)
# df_test.dropna(inplace=True)

# charges = []
# neutral_smiles = []
# for row in tqdm(df_test.iterrows(), total = len(df_test)):
#     smiles = row[1]['SMILES']
#     mol2 = pybel.readstring('smi', smiles)
#     charge = mol2.charge
#     # print(smiles, charge, deviation)
#     charges.append(charge)
#     if charge == 0:
#         neutral_smiles.append(smiles)

# df_neutral_test = pd.DataFrame(neutral_smiles, columns=['SMILES'])
# df_neutral_test = df_neutral_test.merge(df_test, how='inner')
# df_neutral_test.to_csv('combined_data_test_neutral.csv', index=False)