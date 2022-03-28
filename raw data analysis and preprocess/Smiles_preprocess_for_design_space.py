'''
It is the code the sanititate the smiles of the design space. For the current design space
actually all the smiles are valid. However, for bigger design space this script can be very important
'''
from rdkit import Chem
import pandas as pd

def get_valid_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.Kekulize(mol)
    sm = Chem.MolToSmiles(mol)
    return sm

df = pd.read_csv('./preprocessed_data/filtered_whole_design_space_ratio.csv')

df_smiles = df['SMILES']

smiles_numpy = df_smiles.to_numpy()
empty_list = []

for i in range(len(smiles_numpy)):
    smi = get_valid_smiles(smiles_numpy[i])
    empty_list.append(smi)
valid_df = pd.DataFrame(empty_list, columns=['smiles'])
valid_df.dropna(inplace=True)
valid_df.to_csv('./preprocessed_data/valid_design_space.csv')
print('checkpoint')