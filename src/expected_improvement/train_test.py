import math
import os
import pandas as pd
import time
import rdkit.Chem as Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from tqdm import tqdm
from batch_diversity import cut_and_pred_diversity
from test_index import test_index
import subprocess


train_sh = ['chemprop_train', 
            '--data_path', f'../train_set/train_round4.csv', 
            '--dataset_type', 'regression', 
            '--split_type', 'random', 
            '--split_sizes', '0.94', '0.05', '0.01', 
            '--save_dir', f'../model/round_4', 
            '--epochs', '50', 
            '--ensemble_size', '5', 
            '--seed', '666', 
            '--smiles_column', 'SMILES', 
            '--target_columns', 'xTB_S1', 'xTB_T1', 
            '--num_workers', '0']
a = subprocess.check_output(train_sh, shell=False)
print(a)
if a == 0:
    print(f'chemprop_train for round4 has been finished')
time.sleep(10)

test_index(4, f'../model/round_4')
print(f'test_performance for round4 has been finished')