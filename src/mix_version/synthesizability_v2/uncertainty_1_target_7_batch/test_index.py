def test_index(round, check_point_path):
    import os
    import numpy as np
    import pandas as pd

    filePath = '../test_set' 
    os.mkdir(f'../pred_set/round_{round}')
    test_csv = os.listdir(filePath)
    test_result_list = []

    for i in test_csv:
        test_path = filePath + '/' + i
        test_set = pd.read_csv(test_path)
        pred_path = f'../pred_set/round_{round}/{i}'
        os.system(f'chemprop_predict --test_path {test_path} --checkpoint_dir {check_point_path} --preds_path {pred_path} --smiles_column SMILES --ensemble_variance --num_workers 0')
        
        pred_set = pd.read_csv(pred_path).dropna()
        pred_set.reset_index(drop = True, inplace = True)
        pred_set = pred_set[~pred_set['xTB_S1'].isin(['Invalid SMILES'])]
        S1_uncertainty = pred_set['xTB_S1_ensemble_uncal_var'].apply(lambda x: float(x))
        T1_uncertainty = pred_set['xTB_T1_ensemble_uncal_var'].apply(lambda x: float(x))

        # calculate
        S1_error = abs(test_set['xTB_S1'] - pred_set['xTB_S1'])
        T1_error = abs(test_set['xTB_T1'] - pred_set['xTB_T1'])
        S1_spearman_correlation = S1_uncertainty.corr(S1_error,'spearman')
        T1_spearman_correlation = T1_uncertainty.corr(T1_error,'spearman')
        S1_pearson_correlation = S1_uncertainty.corr(S1_error,'pearson')
        T1_pearson_correlation = T1_uncertainty.corr(T1_error,'pearson')
        S1_mae = S1_error.mean()
        T1_mae = T1_error.mean()
        S1_rmse = ((S1_error*S1_error).mean()) ** 0.5
        T1_rmse = ((T1_error*T1_error).mean()) ** 0.5

        # store the result in dictionary
        test_dict = {'File_name' : i, 'S1_spearman_correlation' : S1_spearman_correlation, 
        'T1_spearman_correlation' : T1_spearman_correlation, 'S1_pearson_correlation' : S1_pearson_correlation,
        'T1_pearson_correlation' : T1_pearson_correlation, 'S1_mae' : S1_mae, 'T1_mae' : T1_mae, 'S1_rmse' : S1_rmse, 'T1_rmse' : T1_rmse}
        test_result_list.append(test_dict)

    store_path = f'../test_performance/test_round{round}.csv'
    pd.DataFrame(test_result_list).to_csv(store_path)