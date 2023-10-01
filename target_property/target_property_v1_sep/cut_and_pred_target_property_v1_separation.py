def cut_and_pred_target_v1_separation(dt_tot_csv, cut_len, round, check_point_path, decay_para):
    import math
    import os
    import pandas as pd
    import time
    import numpy as np
    import rdkit.Chem as Chem
    from rdkit.Chem import AllChem
    from rdkit import DataStructs
    from tqdm import tqdm
    import scipy.stats as st

    os.mkdir(f'../cut_data/origin_cut/round_{round}')
    os.mkdir(f'../cut_data/pred_cut/round_{round}')

    dt_tot = pd.read_csv(dt_tot_csv)
    dt_tot.reset_index(drop = True, inplace = True)
    tot_len = len(dt_tot)
    file_num = math.ceil(tot_len / cut_len)
    for i in range(file_num):
        if i != (file_num - 1):
            dt_tot.iloc[i*cut_len:(i+1)*cut_len, :].to_csv(f'../cut_data/origin_cut/round_{round}/dt_tot_{i}.csv')
        if i == (file_num - 1):
            dt_tot.iloc[i*cut_len:, :].to_csv(f'../cut_data/origin_cut/round_{round}/dt_tot_{i}.csv')
    for i in range(file_num):
        print(i)
        test_path = f'../cut_data/origin_cut/round_{round}/dt_tot_{i}.csv'
        pred_path = f'../cut_data/pred_cut/round_{round}/dt_tot_{i}.csv'
        pred = f'dt_tot_{i}.csv'
        k = 0
        while pred not in os.listdir(f'../cut_data/pred_cut/round_{round}'):
            os.system(f'chemprop_predict --test_path {test_path} --checkpoint_dir {check_point_path} --preds_path {pred_path} --smiles_column SMILES --ensemble_variance')
            time.sleep(10)
            if k >= 1:
                print(f'data{i} fail {k} time(s)')
            k = k + 1
            
    dt_list = []
    for i in range(file_num):
        dt_path = f'../cut_data/pred_cut/round_{round}/dt_tot_{i}.csv'
        dt = pd.read_csv(dt_path)
        dt_list.append(dt)
    pred_tot = pd.concat(dt_list).dropna()
    pred_tot.reset_index(drop = True, inplace = True)
    pred_tot = pred_tot[~pred_tot['xTB_S1'].isin(['Invalid SMILES'])]
    E_S1 = pred_tot['xTB_S1'].apply(lambda x: float(x))
    E_T1 = pred_tot['xTB_T1'].apply(lambda x: float(x))
    ratio = E_S1 / E_T1
    uncertainty_tot = decay_para * (pred_tot['xTB_S1_ensemble_uncal_var'].apply(lambda x: float(x)) + pred_tot['xTB_T1_ensemble_uncal_var'].apply(lambda x: float(x))) 
    # decay_para is a factor considering the uncertainty value is so low
    pred_tot['A_score_emitter'] = [st.norm.cdf(2.2, loc=ratio[i], scale=uncertainty_tot[i]) - st.norm.cdf(1.8, loc=ratio[i], scale=uncertainty_tot[i]) for i in range(len(ratio))]
    pred_tot['A_score_sensitizer'] = [st.norm.cdf(1.25, loc=ratio[i], scale=uncertainty_tot[i]) - st.norm.cdf(1, loc=ratio[i], scale=uncertainty_tot[i]) for i in range(len(ratio))] 
    pred_tot.sort_values(by = 'A_score_emitter', ascending = False, inplace = True)
    emitter_index = pred_tot.index[:10000]
    pred_tot.sort_values(by = 'A_score_sensitizer', ascending = False, inplace = True)
    sensitizer_index= pred_tot.index[:10000]
    pred_index = set(emitter_index) | set(sensitizer_index)
    newtrain = dt_tot.iloc[list(pred_index), :]
    train_set = pd.read_csv(f'../train_set/train_round{round}.csv')
    train_set = pd.concat([newtrain, train_set])
    train_set.to_csv(f'../train_set/train_round{round+1}.csv')

    dt_tot_new = dt_tot.iloc[list(set(pred_tot.index) - pred_index), :]
    dt_tot_new.reset_index(drop = True, inplace = True)
    dt_tot_new.to_csv(f'../data_tot/dt_tot_round{round+1}.csv')


