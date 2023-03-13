import pandas as pd
import torch
from tqdm import tqdm
from rdkit import Chem
import numpy as np


def make_mol(s: str, keep_h: bool, add_h: bool, keep_atom_map: bool):
    """
    Builds an RDKit molecule from a SMILES string.

    :param s: SMILES string.
    :param keep_h: Boolean whether to keep hydrogens in the input smiles. This does not add hydrogens, it only keeps them if they are specified.
    :param add_h: Boolean whether to add hydrogens to the input smiles.
    :param keep_atom_map: Boolean whether to keep the original atom mapping.
    :return: RDKit molecule.
    """
    params = Chem.SmilesParserParams()
    params.removeHs = not keep_h if not keep_atom_map else False
    mol = Chem.MolFromSmiles(s, params)

    if add_h:
        mol = Chem.AddHs(mol)

    if keep_atom_map and mol is not None:
        atom_map_numbers = tuple(atom.GetAtomMapNum() for atom in mol.GetAtoms())
        for idx, map_num in enumerate(atom_map_numbers):
            if idx + 1 != map_num:
                new_order = np.argsort(atom_map_numbers).tolist()
                return Chem.rdmolops.RenumberAtoms(mol, new_order)

    return

Total_label = [('Pb', 2), ('Sr', 0), ('Si', 3), ('Au', 0), ('V', 0), ('W', 0), ('Bi', 2), ('Po', 0), ('Mg', 0), ('Be', 0), ('Pd', 0), ('N', 1), ('Se', 1), ('Zr', 0), ('P', 1), ('Tm', 0), ('As', 1), ('V', 2), ('Al', 1), ('Ru', 1), ('I', 1), ('S', 1), ('Ag', 0), ('Si', -1), ('Zr', 2), ('Te', 0), ('I', 3), ('Na', 0), ('Si', 0), ('Eu', 0), ('Sn', 1), ('Br', 2), ('Mo', 1), ('Fe', 0), ('Se', -1), ('O', 1), ('Cu', 0), ('As', -1), ('Al', -1), ('Al', -2), ('Sn', 3), ('Ti', 0), ('S', -1), ('Ca', 0), ('Se', 0), ('At', 0), ('Sb', 1), ('As', 0), ('W', 1), ('Al', 0), ('Ru', 0), ('S', 0), ('Cl', 1), ('Co', 0), ('Al', 2), ('C', 1), ('Hf', 0), ('F', 0), ('Hg', 1), ('Ni', 0), ('Ge', 0), ('Cl', 3), ('N', -1), ('O', -1), ('Si', 2), ('P', -1), ('P', -2), ('Br', 1), ('Mo', 0), ('Tc', 5), ('Xe', 0), ('I', -1), ('N', 0), ('O', 0), ('Sb', -1), ('Al', -3), ('Zn', 0), ('P', 0), ('Mo', 2), ('Ti', 2), ('Fe', 1), ('U', 0), ('Mg', 1), ('I', 0), ('Sb', 0), ('Tl', 0), ('C', -1), ('H', 0), ('Ga', -1), ('Tc', 0), ('Sn', -1), ('Hg', -1), ('In', -1), ('Ir', 0), ('Li', 0), ('Cl', 0), ('I', 2), ('C', 0), ('B', -1), ('Sn', 0), ('Ga', 0), ('Te', 1), ('Hg', 0), ('Os', 0), ('Yb', 0), ('Bi', -1), ('Hf', 2), ('In', 0), ('Pr', 0), ('Cl', 2), ('Pb', 0), ('Cd', 0), ('Cr', 0), ('Si', 1), ('Re', 0), ('K', 0), ('Br', 0), ('Mn', 0), ('W', -1), ('B', 0), ('Rh', 0), ('Sn', 2), ('Y', 0), ('Tc', 4), ('Bi', 0), ('Pt', 0), ('Nb', 0)]


Complex_label = [(0, 'P', 'O', 'N', 'C'), (0, 'N', 'C', 'Si'), (1, 'O', -1), (0, 'O', 'Cl', 'C', 'F'), (0, 'F', 'C', 'B'), (0, 'N', 'C', -1), (0, 'N', 'O', -1), (0, 1, 'O', 'N', 'C', -1), (0, 'Cl', 'B'), (0, 'N', 'B'), (0, 'Cl', 'S', 'N', 'Br'), (0, 1, 'N', 'C', 'F'), (0, 'Cl', 'C', 'O'), (0, 'N', 'Br'), (0, 'P', 'C', 'S'), ('Ge', 0), (0, 'Cl', 'C', 'B'), (0, 'Ge', 'C'), (0, 'P', 'C', 'F'), (0, 1, 'B', 'C', -1), (0, 'Cl', 'C', 'Br'), (0, 'S', 'N', 'C', 'Br'), (0, 'F', 'S'), (0, 'Si'), (0, 1, 'S'), (0, 1, 'O', 'C', -1), (0, 1, 'O', 'C'), (0, 'C', 'S'), (0, 'N', 'O', 'Br'), (0, 'N', 'C', 'P'), (0, 'N', 'C', 'Br', -1), (0, 'O', 'C', 'B'), (0, 'O', 'C', 'Br'), (0, 1, 'P', 'C'), (0, 'O', 'S'), (0, 'Cl', 'O', 'S'), (1, 'O', -1, 'S'), (0, 'As'), (1, 'C', 'S', -1), (0, 'F', 'C', -1), (0, 'P', 'C', 'O'), (0, 'S', -1), (0, 'N', 'Si'), (0, 'Cl', 'S', 'N'), (0, 'N', -1), (0, 1, 'O', 'S', 'N', 'C'), (0, 'Se', 'C', 'O'), (0, 'Cl', 'C', 'Si'), (0, 'O', 'N', 'C', 'F'), (0, 1, 'O', 'C', 'Br'), (0, 'F', 'O', -1), (0, 'Cl', 'C', -1), (0, 1, 'S', 'N', 'C', -1), (0, 'F', 'O'), (0, 1, 'N', 'C', 'F', -1), (0, 1, 'S', 'N', -1), (0, 'N', 'O', 'Cl'), (0, 'F', 'C', 'O'), (0, 1, 'O', 'Cl', 'C'), (0, 'Cl', 'O'), (0, 1, 'O', 'N', 'C'), (0, 'O', 'C', 'N'), (0, 'N', 'O'), (0, 'C', 'B'), (0, 1, 'N', 'O'), (0, 'O', 'Cl', 'S', 'N'), (0, 'N', -1, 'S'), (0, 'Cl', 'N', 'C', -1), (0, 'O', 'C', 'Cl'), (1, 'O', 'S', 'N', -1), ('Cl', -1), (0, 'S', 'B'), (0, 'C', 'Br'), (0, 'O', 'C', 'Si'), (0, 'F', 'C', 'Br'), (0, 'P', 'O'), (0, 'O', 'C', -1), (0, 'O', -1, 'C'), (0, 1, 'N', 'F', -1), (0, 'O', 'B'), (0, 'N', 'C', 'Se'), (0, 'N'), (0, 'O', 'Br'), (0, 'P', 'C', 'Cl'), (0, 1, 'N', 'C', -1), (0, 'Se', 'C'), (0, 'Cl'), (0, 'Se', 'C', 'N'), (0, 'O', 'C', 'F', -1), (0, 'P', 'C', -1), (0, 'O', 'N', 'C', 'Br'), (0, 'C', -1, 'S'), (1, 'O', 'N', 'C', -1), (0, 1, 'N', 'C', 'Br', -1), (0, 'N', 'O', 'P'), (0, 'N', 'F', 'S'), (0, 1, 'O', 'Cl', 'N', 'C'), (0, 'O', 'S', 'C', -1), (0, 'F', 'C', 'N'), (0, 'F', 'S', 'N'), (0, 1, 'B', 'N', 'C', -1), (0, 'P', 'O', -1), (0, 'Cl', 'N'), (0, 'O', 'S', 'N', -1), (0, 'F', 'C', 'Cl'), (0, 'Se'), (0, 'O', 'C', 'P'), (0, 'N', 'O', 'C'), (0, 'C', 'Si'), (0, 'F', 'C', 'Si'), (0, 'Br', 'O', 'C'), (0, 'N', 'Cl'), (0, 'C', -1), (0, 'N', 'C', 'F'), (0, 'Cl', 'O', 'N'), (0, 'Cl', 'C', 'N'), (0, 1, 'N', -1), (0, 'O', 'S', 'C'), (0, 'F', 'O', 'N'), (0, 'O', 'Si', 'N', 'C'), (0, 'O', 'Cl', 'S', 'C'), (0, 'F', 'O', 'Cl'), (0, 1, 'O', 'S', 'C'), (0, 'O', 'Si'), (0, 'O', 'Cl', 'N', 'C'), (0, 'P', -1), (0, 'O', -1), (0, 'P'), (0, 1, 'O'), ('S', -1), (0, 'C', 'As'), (0, 'O', 'N', 'C', -1), (0, 'C'), (0, 'C', -1, 'B'), (1, 'H'), (0, 1, 'O', 'C', 'F', -1), (0, 'C', -1, 'Br'), (0, 1, 'Cl', 'N', 'C', -1), (0, 'B', 'C', 'F', -1), (0, 'O', 'Si', 'C'), (0, 'F', 'C', 'P'), (0, 'O', 'Cl', 'Si', 'C'), (0, 'O', 'S', 'N', 'C'), (0, 1, 'Cl', 'N', -1), (0, 'N', 'P'), (0, 'P', 'C', 'N'), (0, 'Cl', 'S', 'N', 'C'), (0, 'Cl', 'C'), (0, 'N', 'S', 'Br'), (0, 'Cl', 'C', 'P'), (0, 'N', 'C'), (1, 'N', 'O', -1), (0, 'C', 'Si', 'S'), (0, 'O', 'B', 'C'), (0, 'O', 'S', 'Si', 'C'), (1, 'N', 'C', -1), (0, 'F', 'O', 'C'), (0, 'Cl', 'O', 'C'), (0, 1, 'O', 'B', 'C'), (1, 'O', -1, 'C'), (0, 'F', 'N'), (0, 1, 'N'), (0, 1, 'O', 'S'), (0, 1, 'S', 'N', 'C'), (0, 1, 'C', 'S'), (0, 'S', 'N', 'C', 'F'), (0, 'C', 'S', 'B'), (0, 'N', 'C', 'S'), (0, 'N', 'O', 'S'), (0, 'C', 'S', 'Br'), (1, 'N', -1), (0, 'N', 'O', 'F'), (0, 'P', 'N'), (0, 'F', -1, 'N'), (0, 1, 'O', 'N', 'C', 'F'), (0, 'N', 'S', 'Cl'), (0, 'O', -1, 'B'), (0, 'O', 'C', 'F'), (0, 'S', 'C', 'F', -1), (0, 'N', 'S', -1), (0, 'N', 'Se'), (0, 'P', 'O', 'Cl', 'C'), (0, 'B', 'N', 'C', -1), (0, 'S'), (0, 1, 'O', 'C', 'F'), (0, 1, 'P'), ('Ge', 0, 'C'), (0, 'F'), ('O', -1), (0, 1, 'C', 'B'), (0, 'N', 'C', 'Br', 'F'), (0, 'C', 'S', 'Si'), (0, 'F', 'C'), (0, 'N', 'C', 'O'), (0, 1, 'F', 'C'), (0, 1, 'C'), (0, 'P', 'O', 'C'), (0, 'O', 'B', 'N', 'C'), (0, 'C', 'S', -1), (0, 1, 'N', 'P'), (0, 'P', 'O', 'C', -1), (0, 'N', 'C', 'B'), (0, 'N', 'O', 'B'), (0, 'F', 'C', 'S'), (0, 'N', 'C', 'Br'), (0, 'Cl', 'S'), (0, 1, 'N', 'C'), (0, 'N', 'S'), (0, 'N', 'F'), (0, 'P', -1, 'N'), (0, 'P', 'C'), (0, 'O', 'C'), (0, 'Cl', 'C', 'S'), (0, 'O', 'Cl', 'S', 'N', 'C'), (0, 'Cl', 'C', 'F'), (0, 'Cl', 'O', 'F'), (0, 'O'), (0, 'O', 'Si', 'S', 'C'), (0, 'B'), (0, 1, 'B', 'N', 'C'), (1, 'C', -1), (1, 'S', -1), (0, 'Br'), (0, 'S', 'N', 'C', -1), (0, 1, 'Si', 'N', 'C'), (0, 1, 'C', 'Si'), (0, 'Cl', 'N', 'C', 'F'), (0, 'P', 'O', 'S', 'C'), (0, 'O', 'C', 'S'), (0, 1, 'C', -1), (0, 'N', 'C', 'Cl')]


def generate_mask_atom_of_mol(smiles, mask_percentage):

    mol = make_mol(smiles, False, False)
    atom_list = [atom for atom in mol.GetAtoms()]
    masked_atom_index = torch.randperm(len(atom_list))[:int(len(atom_list) * mask_percentage)].tolist()
    sorted_masked_atom_index = list(sorted(masked_atom_index))
    atom_list_np = np.array(atom_list)

    masked_atom_list = list(atom_list_np[sorted_masked_atom_index])
    label_list = []
    for atom in masked_atom_list:

        label_list.append(Total_label.index((atom.GetSymbol(), atom.GetFormalCharge())))


    return sorted_masked_atom_index, label_list


def generate_label_of_mol(smiles, vocab, sorted_masked_atom_index):
    return None

def get_vocab(df):


    df_np = df.to_numpy()

    atom_list = []
    strange_list = []
    for i in tqdm(range(len(df_np))):
        try:
            mol = make_mol(df_np[i], False, False)
            for atom in mol.GetAtoms():
                atom_list.append((atom.GetSymbol(), atom.GetFormalCharge()))
                if abs(atom.GetFormalCharge()) > 4:
                    strange_list.append((df_np, atom.GetSymbol(), atom.GetFormalCharge()))
            atom_list = list(set(atom_list))
        except Exception:
            pass

    atom_set = list(set(atom_list))
    return atom_set, strange_list


def get_vocab_v2(df):
    df_np = df.to_numpy()
    atom_list = []
    strange_list = []
    for i in tqdm(range(len(df_np))):
        try:
            mol = make_mol(df_np[i], False, False)
            for atom in mol.GetAtoms():
                neighbor_charge = [x.GetFormalCharge() for x in atom.GetNeighbors()]
                neighbor = [neigh for neigh in atom.GetNeighbors()]
                second_neighbor = [x.GetNeighbors() for x in neighbor]
                second_symbol = []
                for tup in second_neighbor:
                    for nei in tup:
                        second_symbol.append(nei.GetSymbol())
                neighbor_charge.extend(second_symbol)
                temp = []
                temp.append(atom.GetSymbol())
                temp.append(atom.GetFormalCharge())
                for charge in neighbor_charge:
                    temp.append(charge)
                temp = set(temp)
                temp = tuple(temp)

                atom_list.append(temp)

                if abs(atom.GetFormalCharge()) > 4:
                    strange_list.append((df_np, atom.GetSymbol(), atom.GetFormalCharge()))
            atom_list = list(set(atom_list))
        except Exception:
            pass

    atom_set = list(set(atom_list))
    return atom_set, strange_list


def generate_mask_atom_of_mol_v2(smiles, mask_percentage, vocab):
    mol = make_mol(smiles, False, False)
    atom_list = [atom for atom in mol.GetAtoms()]
    masked_atom_index = torch.randperm(len(atom_list))[:int(len(atom_list) * mask_percentage)].tolist()
    sorted_masked_atom_index = list(sorted(masked_atom_index))
    atom_list_np = np.array(atom_list)

    masked_atom_list = list(atom_list_np[sorted_masked_atom_index])
    label_list = []
    for atom in masked_atom_list:
        neighbor_charge = [x.GetFormalCharge() for x in atom.GetNeighbors()]
        neighbor = [neigh for neigh in atom.GetNeighbors()]
        second_neighbor = [x.GetNeighbors() for x in neighbor]
        second_symbol = []
        for tup in second_neighbor:
            for nei in tup:
                second_symbol.append(nei.GetSymbol())
        neighbor_charge.extend(second_symbol)
        temp = []
        temp.append(atom.GetSymbol())
        temp.append(atom.GetFormalCharge())
        for charge in neighbor_charge:
            temp.append(charge)
        temp = set(temp)
        label_list.append(vocab.index(temp))

    return sorted_masked_atom_index, label_list

data = pd.read_csv('./final_pretrain.csv')
data = data['SMILES']
'''
Get vocab
'''
# vocab_list, strange_list = get_vocab_v2(data)
# vocab_list, strange_list = get_vocab(data)


'''
Save vocab
'''
# file = open('simple_vocab.txt','w')
# # open file for writing
# with open('simple_vocab.txt', 'w') as f:
#     # iterate over list of tuples
#     for item in vocab_list:
#         # convert tuple to comma-separated string
#         item_str = ', '.join(str(i) for i in item)
#         # write the string to a new line in the file
#         f.write(f"{item_str}\n")
#     file.close()
#
# file = open('strange_simple_vocab.txt','w')
# # open file for writing
# with open('strange_simple_vocab.txt', 'w') as f:
#     # iterate over list of tuples
#     for item in strange_list:
#         # convert tuple to comma-separated string
#         item_str = ', '.join(str(i) for i in item)
#         # write the string to a new line in the file
#         f.write(f"{item_str}\n")
#     file.close()
data_np = data.to_numpy()
# print('cool')
'''
Generate pretrain data v1
'''
total_mask_lists = []
total_label_lists = []
vocab = [set(x) for x in Total_label]

for i in tqdm(range(len(data_np))):
    sorted_masked_atom_index, label_list = generate_mask_atom_of_mol(data_np[i], 0.3)
    total_mask_lists.append(sorted_masked_atom_index)
    total_label_lists.append(label_list)
data = pd.DataFrame(data)
data['Masked_index'] = total_mask_lists
data['Label'] = total_label_lists
data.to_csv('./final_pretrain_simple_vocab.csv')
# print('cool')
