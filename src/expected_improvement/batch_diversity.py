def cut_and_pred_target_v1(dt_tot_csv, cut_len, round, check_point_path):
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
    from scipy.stats import norm

    def ratio_variance(t1_pred, s1_pred, t1_var, s1_var):
        return (1 / s1_pred**2) * t1_var + (t1_pred**2 / s1_pred**4) * s1_var
    
    def expected_improvement_abs(mu, sigma, x_nearest, xi=0.05):
        Z = (np.abs(mu - x_nearest) - xi) / sigma
        return (np.abs(mu - x_nearest) - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)

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
    ratio_ts = E_T1 / E_S1
    ratio_st = E_S1 / E_T1
    T1_var = pred_tot['xTB_T1_ensemble_uncal_var'].apply(lambda x: float(x))
    S1_var = pred_tot['xTB_S1_ensemble_uncal_var'].apply(lambda x: float(x))
    ratio_var_ts = ratio_variance(E_T1, E_S1, T1_var, S1_var)
    ratio_std_ts = np.sqrt(ratio_var_ts)
    ratio_var_st = ratio_variance(E_S1, E_T1, S1_var, T1_var)
    ratio_std_st = np.sqrt(ratio_var_st)

    # decay_para is a factor considering the uncertainty value is so low
    pred_tot['ei_values_sens'] = expected_improvement_abs(ratio_ts, ratio_std_ts, 0.9)
    pred_tot['ei_values_emit'] = expected_improvement_abs(ratio_st, ratio_std_st, 2)
    pred_tot.sort_values(by = 'ei_values_sens', ascending = False, inplace = True)
    emitter_index = pred_tot.index[:10000]
    pred_tot.sort_values(by = 'ei_values_emit', ascending = False, inplace = True)
    sensitizer_index= pred_tot.index[:10000]
    pred_index = set(emitter_index) | set(sensitizer_index)
    newtrain = dt_tot.iloc[list(pred_index), :]
    train_set = pd.read_csv(f'../train_set/train_round{round}.csv')
    train_set = pd.concat([newtrain, train_set])
    train_set.to_csv(f'../train_set/train_round{round+1}.csv')

    dt_tot_new = dt_tot.iloc[list(set(pred_tot.index) - pred_index), :]
    dt_tot_new.reset_index(drop = True, inplace = True)
    dt_tot_new.to_csv(f'../data_tot/dt_tot_round{round+1}.csv')


