def pure_uncertainty(dt_tot_csv, cut_len, round, check_point_path):
    import math
    import os
    import pandas as pd
    import time

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
    pred_tot['uncertainty_tot'] = pred_tot['xTB_S1_ensemble_uncal_var'].apply(lambda x: float(x)) + pred_tot['xTB_T1_ensemble_uncal_var'].apply(lambda x: float(x))
    pred_tot.sort_values(by = 'uncertainty_tot', ascending = False, inplace = True)
    pred_index = pred_tot.index[:20000]
    
    newtrain = dt_tot.iloc[pred_index, :]
    train_set = pd.read_csv(f'../train_set/train_round{round}.csv')
    train_set = pd.concat([newtrain, train_set])
    train_set.to_csv(f'../train_set/train_round{round+1}.csv')

    dt_tot_new = dt_tot.iloc[list(set(pred_tot.index)-set(pred_index)), :]
    dt_tot_new.reset_index(drop = True, inplace = True)
    dt_tot_new.to_csv(f'../data_tot/dt_tot_round{round+1}.csv')

def batch_diversity(dt_tot_csv, cut_len, round, check_point_path, similar_value_threshold, similar_num_threshold):
    import math
    import os
    import pandas as pd
    import time
    import rdkit.Chem as Chem
    from rdkit.Chem import AllChem
    from rdkit import DataStructs
    from tqdm import tqdm

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
            os.system(f'chemprop_predict --test_path {test_path} --checkpoint_dir {check_point_path} --preds_path {pred_path} --smiles_column SMILES --ensemble_variance --num_workers 0')
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
    pred_tot['uncertainty_tot'] = pred_tot['xTB_S1_ensemble_uncal_var'].apply(lambda x: float(x)) + pred_tot['xTB_T1_ensemble_uncal_var'].apply(lambda x: float(x))
    pred_tot.sort_values(by = 'uncertainty_tot', ascending = False, inplace = True)

    suggest_list = []
    total_fingerprint = []
    pred_len = len(pred_tot)

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

def target_property_v1_sep_without_diversity(dt_tot_csv, cut_len, round, check_point_path, decay_para):
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

def target_property_v1_with_diversity(dt_tot_csv, cut_len, round, check_point_path, decay_para):
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

    suggest_list = []
    total_fingerprint = []
    pred_len = len(pred_tot)
    similar_value_threshold = 0.5
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
            if len(suggest_list) >= 10000:
                break
            if (len(suggest_list) % 200) == 0:
                print(f'suggest list size is {len(suggest_list)}')
    pred_index_emitter = pred_tot.index[suggest_list]

    pred_tot.sort_values(by = 'A_score_sensitizer', ascending = False, inplace = True)
    suggest_list = []
    total_fingerprint = []
    pred_len = len(pred_tot)
    similar_value_threshold = 0.5
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
            if len(suggest_list) >= 10000:
                break
            if (len(suggest_list) % 200) == 0:
                print(f'suggest list size is {len(suggest_list)}')
    pred_index_sensitizer = pred_tot.index[suggest_list]

    pred_index = set(pred_index_emitter) | set(pred_index_sensitizer)
    newtrain = dt_tot.iloc[list(pred_index), :]
    train_set = pd.read_csv(f'../train_set/train_round{round}.csv')
    train_set = pd.concat([newtrain, train_set])
    train_set.to_csv(f'../train_set/train_round{round+1}.csv')

    dt_tot_new = dt_tot.iloc[list(set(pred_tot.index) - pred_index), :]
    dt_tot_new.reset_index(drop = True, inplace = True)
    dt_tot_new.to_csv(f'../data_tot/dt_tot_round{round+1}.csv')

def target_property_v2_without_diversity(dt_tot_csv, cut_len, round, check_point_path, decay_para):
    import math
    import os
    import pandas as pd
    import time
    import numpy as np
    import rdkit.Chem as Chem
    from rdkit.Chem import AllChem
    from rdkit import DataStructs
    from tqdm import tqdm

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

    pred_tot['A_score_emitter'] = np.exp(-decay_para * np.abs(2-ratio)**2)
    pred_tot['A_score_sensitizer'] = np.exp(-decay_para * np.abs(1/0.9-ratio)**2)
    pred_tot.sort_values(by = 'A_score_emitter', ascending = False, inplace = True)
    emitter_index = pred_tot.index[:10000]
    pred_tot.sort_values(by = 'A_score_sensitizer', ascending = False, inplace = True)
    sensitizer_index= pred_tot.index[:10000]
    pred_index = set(emitter_index) | set(sensitizer_index)
    newtrain = dt_tot.iloc[list(pred_index), :]

    train_set = pd.read_csv(f'../train_set/train_round{round}.csv')
    train_set = pd.concat([newtrain, train_set])
    train_set.to_csv(f'../train_set/train_round{round+1}.csv')

    dt_tot_new = dt_tot.iloc[list(set(pred_tot.index)-set(pred_index)), :]
    dt_tot_new.reset_index(drop = True, inplace = True)
    dt_tot_new.to_csv(f'../data_tot/dt_tot_round{round+1}.csv')

def target_property_v2_with_diversity(dt_tot_csv, cut_len, round, check_point_path, decay_para):
    import math
    import os
    import pandas as pd
    import time
    import numpy as np
    import rdkit.Chem as Chem
    from rdkit.Chem import AllChem
    from rdkit import DataStructs
    from tqdm import tqdm

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

    pred_tot['A_score_emitter'] = np.exp(-decay_para * np.abs(2-ratio)**2)
    pred_tot['A_score_sensitizer'] = np.exp(-decay_para * np.abs(1/0.9-ratio)**2)

    pred_tot.sort_values(by = 'A_score_emitter', ascending = False, inplace = True)
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
            if len(suggest_list) >= 10000:
                break
            if (len(suggest_list) % 200) == 0:
                print(f'suggest list size is {len(suggest_list)}')
    pred_index_emitter = pred_tot.index[suggest_list]

    pred_tot.sort_values(by = 'A_score_sensitizer', ascending = False, inplace = True)
    suggest_list = []
    total_fingerprint = []

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
            if len(suggest_list) >= 10000:
                break
            if (len(suggest_list) % 200) == 0:
                print(f'suggest list size is {len(suggest_list)}')
    pred_index_sensitizer = pred_tot.index[suggest_list]

    pred_index = set(pred_index_emitter) | set(pred_index_sensitizer)
    newtrain = dt_tot.iloc[list(pred_index), :]

    train_set = pd.read_csv(f'../train_set/train_round{round}.csv')
    train_set = pd.concat([newtrain, train_set])
    train_set.to_csv(f'../train_set/train_round{round+1}.csv')

    dt_tot_new = dt_tot.iloc[list(set(pred_tot.index)-set(pred_index)), :]
    dt_tot_new.reset_index(drop = True, inplace = True)
    dt_tot_new.to_csv(f'../data_tot/dt_tot_round{round+1}.csv')

def expected_improvement(dt_tot_csv, cut_len, round, check_point_path):
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
    
    def expected_improvement_abs(mu, sigma, x_nearest, xi=0.01):
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

def uncertainty_target_mix(dt_tot_csv, cut_len, round, check_point_path):
    import math
    import os
    import pandas as pd
    import time
    import numpy as np
    import rdkit.Chem as Chem
    from rdkit.Chem import AllChem
    from rdkit import DataStructs
    from tqdm import tqdm

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

    pred_tot['A_score_emitter'] = np.abs(2-ratio)
    pred_tot['A_score_sensitizer'] = np.abs(1/0.9-ratio)
    pred_tot['uncertainty_tot'] = pred_tot['xTB_S1_ensemble_uncal_var'].apply(lambda x: float(x)) + pred_tot['xTB_T1_ensemble_uncal_var'].apply(lambda x: float(x))
    columns_to_normalize = ['A_score_emitter', 'A_score_sensitizer', 'uncertainty_tot']
    pred_tot[columns_to_normalize] = (pred_tot[columns_to_normalize] - pred_tot[columns_to_normalize].mean()) / pred_tot[columns_to_normalize].std()
    pred_tot['mix_metric_emitter'] = pred_tot['A_score_emitter'] + pred_tot['uncertainty_tot']
    pred_tot['mix_metric_sensitizer'] = pred_tot['A_score_sensitizer'] + pred_tot['uncertainty_tot']

    pred_tot.sort_values(by = 'mix_metric_emitter', ascending = False, inplace = True)
    suggest_list = []
    total_fingerprint = []
    pred_len = len(pred_tot)
    similar_value_threshold = 0.5
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
            if len(suggest_list) >= 10000:
                break
            if (len(suggest_list) % 200) == 0:
                print(f'suggest list size is {len(suggest_list)}')
    pred_index_emitter = pred_tot.index[suggest_list]

    pred_tot.sort_values(by = 'mix_metric_sensitizer', ascending = False, inplace = True)
    suggest_list = []
    total_fingerprint = []

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
            if len(suggest_list) >= 10000:
                break
            if (len(suggest_list) % 200) == 0:
                print(f'suggest list size is {len(suggest_list)}')
    pred_index_sensitizer = pred_tot.index[suggest_list]

    pred_index = set(pred_index_emitter) | set(pred_index_sensitizer)
    newtrain = dt_tot.iloc[list(pred_index), :]

    train_set = pd.read_csv(f'../train_set/train_round{round}.csv')
    train_set = pd.concat([newtrain, train_set])
    train_set.to_csv(f'../train_set/train_round{round+1}.csv')

    dt_tot_new = dt_tot.iloc[list(set(pred_tot.index)-set(pred_index)), :]
    dt_tot_new.reset_index(drop = True, inplace = True)
    dt_tot_new.to_csv(f'../data_tot/dt_tot_round{round+1}.csv')

def substructure(dt_tot_csv, cut_len, round, check_point_path):
    import math
    import os
    import pandas as pd
    import time
    from tqdm import tqdm
    from rdkit import Chem

    os.mkdir(f'../cut_data/origin_cut/round_{round}')
    os.mkdir(f'../cut_data/pred_cut/round_{round}')
    substructure_emit = pd.read_csv('emit_substructure.csv')['smiles']
    substructure_sens = pd.read_csv('sens_substructure.csv')['smiles']
    substructure_tot = pd.concat([substructure_emit, substructure_sens])
    
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
    pred_tot['uncertainty_tot'] = pred_tot['xTB_S1_ensemble_uncal_var'].apply(lambda x: float(x)) + pred_tot['xTB_T1_ensemble_uncal_var'].apply(lambda x: float(x))
    pred_tot.sort_values(by = 'uncertainty_tot', ascending = False, inplace = True)

    suggest_list = []
    pred_len = len(pred_tot)
    smiles_tot = pred_tot['SMILES']

    for i in tqdm(range(pred_len)):
        mol = Chem.MolFromSmiles(smiles_tot[i])
        for sub in substructure_tot:
            sub_mol = Chem.MolFromSmiles(sub)
            if mol.HasSubstructMatch(sub_mol):
                suggest_list.append(i)
                break
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










