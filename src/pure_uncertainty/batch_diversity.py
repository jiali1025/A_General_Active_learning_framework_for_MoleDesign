def cut_and_pred_diversity(dt_tot_csv, cut_len, round, check_point_path):
    import math
    import os
    import pandas as pd
    import time
    import rdkit.Chem as Chem
    from rdkit.Chem import AllChem
    from rdkit import DataStructs
    from tqdm import tqdm

    os.mkdir(f'../batch_version/cut_data/origin_cut/round_{round}')
    os.mkdir(f'../batch_version/cut_data/pred_cut/round_{round}')

    dt_tot = pd.read_csv(dt_tot_csv)
    dt_tot.reset_index(drop = True, inplace = True)
    tot_len = len(dt_tot)
    file_num = math.ceil(tot_len / cut_len)
    for i in range(file_num):
        if i != (file_num - 1):
            dt_tot.iloc[i*cut_len:(i+1)*cut_len, :].to_csv(f'../batch_version/cut_data/origin_cut/round_{round}/dt_tot_{i}.csv')
        if i == (file_num - 1):
            dt_tot.iloc[i*cut_len:, :].to_csv(f'../batch_version/cut_data/origin_cut/round_{round}/dt_tot_{i}.csv')
    for i in range(file_num):
        print(i)
        test_path = f'../batch_version/cut_data/origin_cut/round_{round}/dt_tot_{i}.csv'
        pred_path = f'../batch_version/cut_data/pred_cut/round_{round}/dt_tot_{i}.csv'
        pred = f'dt_tot_{i}.csv'
        k = 0
        while pred not in os.listdir(f'../batch_version/cut_data/pred_cut/round_{round}'):
            os.system(f'chemprop_predict --test_path {test_path} --checkpoint_dir {check_point_path} --preds_path {pred_path} --smiles_column SMILES --ensemble_variance --num_workers 0')
            time.sleep(10)
            if k >= 1:
                print(f'data{i} fail {k} time(s)')
            k = k + 1
            
    dt_list = []
    for i in range(file_num):
        dt_path = f'../batch_version/cut_data/pred_cut/round_{round}/dt_tot_{i}.csv'
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
    loop = 0
    similar_value_threshold = 0.4
    similar_num_threshold = 5

    total_smiles = pred_tot['SMILES'][0:20000]
    for smile in tqdm(total_smiles):
        mol = Chem.MolFromSmiles(smile)
        fingerprint = AllChem.GetMorganFingerprint(mol)
        total_fingerprint.append(fingerprint)

    for i in tqdm(range(20000)):
        suggest_list.append(i)
        k = 0
        loop += 1
        query_fingerprint = total_fingerprint[i]
        if i >= 10:
            target_fingerprints = total_fingerprint[0:i]
            scores = DataStructs.BulkTanimotoSimilarity(query_fingerprint, target_fingerprints)
            total_similar_num = len(list(filter(lambda x: x > similar_value_threshold, scores)))
            if total_similar_num > similar_num_threshold:
                suggest_list.pop()
    
    if len(suggest_list) < 20000:
        for i in tqdm(range(20000, pred_len)):
            new_smile = pred_tot['SMILES'][i]
            new_mol = Chem.MolFromSmiles(new_smile)
            new_fingerprint = AllChem.GetMorganFingerprint(new_mol,2)
            total_fingerprint.append(new_fingerprint)
            suggest_list.append(i)
            query_fingerprint = total_fingerprint[i]
            target_fingerprints = total_fingerprint[0:i]
            scores = DataStructs.BulkTanimotoSimilarity(query_fingerprint, target_fingerprints)
            total_similar_num = len(list(filter(lambda x: x > similar_value_threshold, scores)))
            if total_similar_num > similar_num_threshold:
                suggest_list.pop()
            if len(suggest_list) >= 20000:
                break

    pred_index = pred_tot.index[suggest_list]
    newtrain = dt_tot.iloc[pred_index, :]
    train_set = pd.read_csv(f'../batch_version/train_set/train_round{round}.csv')
    train_set = pd.concat([newtrain, train_set])
    train_set.to_csv(f'../batch_version/train_set/train_round{round+1}.csv')

    dt_tot_new = dt_tot.iloc[list(set(pred_tot.index)-set(pred_index)), :]
    dt_tot_new.reset_index(drop = True, inplace = True)
    dt_tot_new.to_csv(f'../batch_version/data_tot/dt_tot_round{round+1}.csv')