import math
import os
import pandas as pd
import time
import rdkit.Chem as Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from tqdm import tqdm
from cut_and_pred_target_property_v2_separation import cut_and_pred_target_property_v2_separation
from test_index import test_index
import subprocess

total_round = 9
start_round = 1

def main():
    for i in range(start_round, total_round):
        check_point_path = f'../model/round_{i}'
        dt_tot = f'../data_tot/dt_tot_round{i}.csv'
        decay_param = 25
        cut_and_pred_target_property_v2_separation(dt_tot, 30000, i, check_point_path, decay_param)
        print(f'cut_and_pred round{i} has been finished')
        time.sleep(10)

        if i == 1: # data_size = 25k
            add_arg = ['--depth', '5', '--ffn_num_layers', '3', '--dropout', '0.05', '--hidden_size', '1600']
        if i == 2: # data_size = 45k
            add_arg = ['--depth', '6', '--ffn_num_layers', '3', '--dropout', '0.25', '--hidden_size', '1000']
        if i == 3: # data_size = 65k
            add_arg = ['--depth', '3', '--ffn_num_layers', '2', '--dropout', '0.05', '--hidden_size', '1900']
        if i == 4: # data_size = 85k
            add_arg = ['--depth', '3', '--ffn_num_layers', '2', '--dropout', '0.05', '--hidden_size', '1900']
        if i == 5: # data_size = 105k
            add_arg = ['--depth', '3', '--ffn_num_layers', '3', '--hidden_size', '500']
        if i == 6: # data_size = 125k
            add_arg = ['--depth', '5', '--ffn_num_layers', '3', '--dropout', '0.05', '--hidden_size', '1600']
        if i == 7: # data_size = 145k
            add_arg = ['--depth', '3', '--ffn_num_layers', '2', '--dropout', '0.05', '--hidden_size', '1900']
        if i == 8: # data_size = 165k
            add_arg = ['--depth', '3', '--ffn_num_layers', '2', '--dropout', '0.05', '--hidden_size', '1900']

        train_sh_const = ['chemprop_train', 
                    '--data_path', f'../train_set/train_round{i+1}.csv', 
                    '--dataset_type', 'regression', 
                    '--split_type', 'random', 
                    '--split_sizes', '0.94', '0.05', '0.01', 
                    '--save_dir', f'../model/round_{i+1}', 
                    '--epochs', '50', 
                    '--ensemble_size', '5', 
                    '--seed', '666', 
                    '--smiles_column', 'SMILES', 
                    '--target_columns', 'xTB_S1', 'xTB_T1']
        train_sh = train_sh_const + add_arg
        a = subprocess.check_output(train_sh, shell=False)
        print(a)
        if a == 0:
            print(f'chemprop_train for round{i+1} has been finished')
        time.sleep(10)

        test_index(i+1, f'../model/round_{i+1}')
        print(f'test_performance for round{i+1} has been finished')

main()

