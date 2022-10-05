import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
import json
import sys
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn import linear_model
from cycler import cycler
import string
from itertools import cycle
import pandas as pd


def label_axes(fig, labels=None, loc=None, **kwargs):
    if labels is None:
        labels = string.ascii_lowercase
    labels = cycle(labels)
    if loc is None:
        loc = (-0.1, 1.1)
    axes = [ax for ax in fig.axes if ax.get_label() != '<colorbar>']
    for ax, lab in zip(axes, labels):
        ax.annotate('(' + lab + ')', size=14, xy=loc,
                    xycoords='axes fraction',
                    **kwargs)


plt.style.use(['science', 'grid'])
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
colors = [colors[3], colors[1], colors[0]]
colors_nipy1 = mpl.cm.nipy_spectral(np.linspace(0.1, 0.9, 6))
colors_nipy2 = mpl.cm.nipy_spectral(np.linspace(0.6, 0.9, 7))
colors_nipy = list(colors_nipy1[0:3]) + list(colors_nipy2[3:-2]) + list(colors_nipy1[-1:])
plt.rcParams['axes.prop_cycle'] = cycler(color=colors)


def gen_train_data():
    df = pd.read_csv('combined_data_train_neutral.csv')
    df['TDDFT_S1_T1'] = df['TDDFT_S1']/df['TDDFT_T1']
    df['xTB_S1_T1'] = df['xTB_S1']/df['xTB_T1']
    df.drop_duplicates(subset = 'SMILES', inplace=True)
    df.dropna(inplace=True)
    df.to_csv('comb_train_data.csv', columns=['SMILES', 'TDDFT_S1', 'TDDFT_T1', 'TDDFT_S1_T1'], index=False)
    df.to_csv('comb_featu_data.csv', columns=['xTB_S1', 'xTB_T1', 'xTB_S1_T1'], index=False)


def train_ML():
    os.system('chemprop_train --data_path comb_train_data.csv --features_path comb_featu_data.csv '
              '--dataset_type regression --save_dir xTB_ML_model_largemol_all_neutral '
              '--target_columns TDDFT_S1 TDDFT_T1 TDDFT_S1_T1 --target_weights 0.2 0.2 0.6 --num_folds 10 --split_type cv-no-test')


# def gen_test_data():
#     datatest = pd.read_csv('combined_data_large_mol_test.csv')
#     datatest['TDDFT_S1_T1'] = datatest['TDDFT_S1']/datatest['TDDFT_T1']
#     datatest['xTB_S1_T1'] = datatest['xTB_S1']/datatest['xTB_T1']
#     datatest.to_csv('comb_test_data_SMILES.csv', columns=['SMILES'], index=False)
#     datatest.to_csv('comb_test_data_features.csv', columns=['xTB_S1', 'xTB_T1', 'xTB_S1_T1'], index=False)


# def predict_ML():
#     os.system('chemprop_predict --test_path comb_test_data_SMILES.csv --features_path comb_test_data_features.csv '
#               '--checkpoint_dir xTB_ML_model_largemol2 --preds_path comb_test_data_preds.csv --drop_extra_columns')


# def evaluate_predictions_3k_largemol():
#     df_data = pd.read_csv('combined_data_large_mol_test.csv')
#     df_data['TDDFT_S1_T1'] = df_data['TDDFT_S1']/df_data['TDDFT_T1']
#     df_data['xTB_S1_T1'] = df_data['xTB_S1']/df_data['xTB_T1']
#     df_preds = pd.read_csv('comb_test_data_preds.csv')
#     df_preds.rename({'TDDFT_S1': 'pred_S1', 'TDDFT_T1': 'pred_T1', 'TDDFT_S1_T1': 'pred_S1_T1'}, axis='columns', inplace=True)
#     df_all = df_preds.merge(df_data, on='SMILES')

#     r2_S1 = r2_score(df_all['TDDFT_S1'], df_all['xTB_S1'])
#     r2_T1 = r2_score(df_all['TDDFT_S1'], df_all['xTB_T1'])
#     MAE_S1 = mean_absolute_error(df_all['TDDFT_S1'], df_all['xTB_S1'])
#     MAE_T1 = mean_absolute_error(df_all['TDDFT_T1'], df_all['xTB_T1'])
#     MAPE_S1 = mean_absolute_percentage_error(df_all['TDDFT_S1'], df_all['xTB_S1'])
#     MAPE_T1 = mean_absolute_percentage_error(df_all['TDDFT_T1'], df_all['xTB_T1'])
#     MAE_ratio = mean_absolute_error(df_all['TDDFT_S1_T1'], df_all['xTB_S1_T1'])
#     MAPE_ratio = mean_absolute_percentage_error(df_all['TDDFT_S1_T1'], df_all['xTB_S1_T1'])
#     # r2_lin_S1 = r2_score(df_all['TDDFT_S1'], df_all['xTB_Lin_S1'])
#     # r2_lin_T1 = r2_score(df_all['TDDFT_T1'], df_all['xTB_Lin_T1'])
#     # MAE_lin_S1 = mean_absolute_error(df_all['TDDFT_S1'], df_all['xTB_Lin_S1'])
#     # MAE_lin_T1 = mean_absolute_error(df_all['TDDFT_T1'], df_all['xTB_Lin_T1'])
#     r2_ML_S1 = r2_score(df_all['TDDFT_S1'], df_all['pred_S1'])
#     r2_ML_T1 = r2_score(df_all['TDDFT_T1'], df_all['pred_T1'])
#     MAE_ML_S1 = mean_absolute_error(df_all['TDDFT_S1'], df_all['pred_S1'])
#     MAE_ML_T1 = mean_absolute_error(df_all['TDDFT_T1'], df_all['pred_T1'])
#     MAPE_ML_S1 = mean_absolute_percentage_error(df_all['TDDFT_S1'], df_all['pred_S1'])
#     MAPE_ML_T1 = mean_absolute_percentage_error(df_all['TDDFT_T1'], df_all['pred_T1'])
#     MAE_ML_ratio = mean_absolute_error(df_all['TDDFT_S1_T1'], df_all['pred_S1_T1'])
#     MAPE_ML_ratio = mean_absolute_percentage_error(df_all['TDDFT_S1_T1'], df_all['pred_S1_T1'])
#     print('3k large mol dataset')
#     print('stda')
#     print('r2_S1: ', r2_S1)
#     print('r2_T1: ', r2_T1)
#     print('MAE_S1: ', MAE_S1)
#     print('MAE_T1: ', MAE_T1)
#     print('MAPE_S1: ', MAPE_S1)
#     print('MAPE_T1: ', MAPE_T1)
#     print('MAE_ratio: ', MAE_ratio)
#     print('MAPE_ratio: ', MAPE_ratio)
#     print('ML')
#     print('r2_S1: ', r2_ML_S1)
#     print('r2_T1: ', r2_ML_T1)
#     print('MAE_S1: ', MAE_ML_S1)
#     print('MAE_T1: ', MAE_ML_T1)
#     print('MAPE_S1: ', MAPE_ML_S1)
#     print('MAPE_T1: ', MAPE_ML_T1)
#     print('MAE_ratio: ', MAE_ML_ratio)
#     print('MAPE_ratio: ', MAPE_ML_ratio)

#     fig = plt.figure(num=1, clear=True, figsize=[7, 4],  dpi=300)

#     ax = fig.add_subplot(121)
#     plt.plot(df_all['xTB_S1'], df_all['TDDFT_S1'], '.', color=colors[0], markersize=1, label='xTB-sTDA')
#     plt.plot(df_all['pred_S1'], df_all['TDDFT_S1'], '.', color=colors[2], markersize=1, label='xTB-ML')
#     x = np.linspace(0, 9, 100)
#     plt.plot(x, x, 'k--')
#     plt.xlim(0, 9)
#     plt.ylim(0, 9)
#     plt.gca().set_aspect('equal', adjustable='box')
#     plt.xlabel('xTB-sTDA S1 (eV)')
#     plt.ylabel('TDDFT S1 (eV)')
#     plt.legend()
#     plt.annotate('R2 orig: %0.2f\n' % r2_S1 +
#                  #  'R2 lin: %0.2f\n' % r2_lin_S1 +
#                  'R2 ML: %0.2f\n' % r2_ML_S1 +
#                  'MAE orig: %0.2f\n' % MAE_S1 +
#                  #  'MAE lin: %0.2f\n' % MAE_lin_S1 +
#                  'MAE ML: %0.2f' % MAE_ML_S1,
#                  (8.5, 0.5),
#                  bbox=dict(facecolor='white', alpha=0.5),
#                  ha='right')
#     plt.tight_layout()

#     ax = fig.add_subplot(122)
#     plt.plot(df_all['xTB_T1'], df_all['TDDFT_T1'], '.', color=colors[0], markersize=1, label='xTB-sTDA')
#     plt.plot(df_all['pred_T1'], df_all['TDDFT_T1'], '.', color=colors[2], markersize=1, label='xTB-ML')
#     plt.plot(x, x, 'k--')
#     plt.xlim(0, 9)
#     plt.ylim(0, 9)
#     plt.gca().set_aspect('equal', adjustable='box')
#     plt.xlabel('xTB-sTDA T1 (eV)')
#     plt.ylabel('TDDFT T1 (eV)')
#     plt.legend()
#     plt.annotate('R2 orig: %0.2f\n' % r2_T1 +
#                  #  'R2 lin: %0.2f\n' % r2_lin_T1 +
#                  'R2 ML: %0.2f\n' % r2_ML_T1 +
#                  'MAE orig: %0.2f\n' % MAE_T1 +
#                  #  'MAE lin: %0.2f\n' % MAE_lin_T1 +
#                  'MAE ML: %0.2f' % MAE_ML_T1,
#                  (8.5, 0.5),
#                  bbox=dict(facecolor='white', alpha=0.5),
#                  ha='right')
#     plt.tight_layout()

#     plt.savefig('large_mol_3ktest_xTB_ML.png')

# def evaluate_predictions_testset():
#     df_data = pd.read_csv('xTB_ML_model_largemol2/fold_0/test_full.csv')
#     df_preds = pd.read_csv('xTB_ML_model_largemol2/test_preds.csv')
#     df_preds.rename({'TDDFT_S1': 'pred_S1', 'TDDFT_T1': 'pred_T1', 'TDDFT_S1_T1': 'pred_S1_T1'}, axis='columns', inplace=True)
#     df_preds['SMILES'] = [x[2:-2] for x in df_preds['smiles']]
#     df_xtb = pd.read_csv('xTB_ML_model_largemol2/fold_0/test_features.csv')
#     df_xtb['SMILES'] = df_data['SMILES']
#     df_all = df_preds.merge(df_data, on='SMILES').merge(df_xtb, on='SMILES')

#     r2_S1 = r2_score(df_all['TDDFT_S1'], df_all['xTB_S1'])
#     r2_T1 = r2_score(df_all['TDDFT_S1'], df_all['xTB_T1'])
#     MAE_S1 = mean_absolute_error(df_all['TDDFT_S1'], df_all['xTB_S1'])
#     MAE_T1 = mean_absolute_error(df_all['TDDFT_T1'], df_all['xTB_T1'])
#     MAPE_S1 = mean_absolute_percentage_error(df_all['TDDFT_S1'], df_all['xTB_S1'])
#     MAPE_T1 = mean_absolute_percentage_error(df_all['TDDFT_T1'], df_all['xTB_T1'])
#     MAE_ratio = mean_absolute_error(df_all['TDDFT_S1_T1'], df_all['xTB_S1_T1'])
#     MAPE_ratio = mean_absolute_percentage_error(df_all['TDDFT_S1_T1'], df_all['xTB_S1_T1'])
#     # r2_lin_S1 = r2_score(df_all['TDDFT_S1'], df_all['xTB_Lin_S1'])
#     # r2_lin_T1 = r2_score(df_all['TDDFT_T1'], df_all['xTB_Lin_T1'])
#     # MAE_lin_S1 = mean_absolute_error(df_all['TDDFT_S1'], df_all['xTB_Lin_S1'])
#     # MAE_lin_T1 = mean_absolute_error(df_all['TDDFT_T1'], df_all['xTB_Lin_T1'])
#     r2_ML_S1 = r2_score(df_all['TDDFT_S1'], df_all['pred_S1'])
#     r2_ML_T1 = r2_score(df_all['TDDFT_T1'], df_all['pred_T1'])
#     MAE_ML_S1 = mean_absolute_error(df_all['TDDFT_S1'], df_all['pred_S1'])
#     MAE_ML_T1 = mean_absolute_error(df_all['TDDFT_T1'], df_all['pred_T1'])
#     MAPE_ML_S1 = mean_absolute_percentage_error(df_all['TDDFT_S1'], df_all['pred_S1'])
#     MAPE_ML_T1 = mean_absolute_percentage_error(df_all['TDDFT_T1'], df_all['pred_T1'])
#     MAE_ML_ratio = mean_absolute_error(df_all['TDDFT_S1_T1'], df_all['pred_S1_T1'])
#     MAPE_ML_ratio = mean_absolute_percentage_error(df_all['TDDFT_S1_T1'], df_all['pred_S1_T1'])
#     print('chemprop 10% testset')
#     print('stda')
#     print('r2_S1: ', r2_S1)
#     print('r2_T1: ', r2_T1)
#     print('MAE_S1: ', MAE_S1)
#     print('MAE_T1: ', MAE_T1)
#     print('MAPE_S1: ', MAPE_S1)
#     print('MAPE_T1: ', MAPE_T1)
#     print('MAE_ratio: ', MAE_ratio)
#     print('MAPE_ratio: ', MAPE_ratio)
#     print('ML')
#     print('r2_S1: ', r2_ML_S1)
#     print('r2_T1: ', r2_ML_T1)
#     print('MAE_S1: ', MAE_ML_S1)
#     print('MAE_T1: ', MAE_ML_T1)
#     print('MAPE_S1: ', MAPE_ML_S1)
#     print('MAPE_T1: ', MAPE_ML_T1)
#     print('MAE_ratio: ', MAE_ML_ratio)
#     print('MAPE_ratio: ', MAPE_ML_ratio)

#     fig = plt.figure(num=1, clear=True, figsize=[7, 4],  dpi=300)

#     ax = fig.add_subplot(121)
#     plt.plot(df_all['xTB_S1'], df_all['TDDFT_S1'], '.', color=colors[0], markersize=1, label='xTB-sTDA')
#     plt.plot(df_all['pred_S1'], df_all['TDDFT_S1'], '.', color=colors[2], markersize=1, label='xTB-ML')
#     x = np.linspace(0, 9, 100)
#     plt.plot(x, x, 'k--')
#     plt.xlim(0, 9)
#     plt.ylim(0, 9)
#     plt.gca().set_aspect('equal', adjustable='box')
#     plt.xlabel('xTB-sTDA S1 (eV)')
#     plt.ylabel('TDDFT S1 (eV)')
#     plt.legend()
#     plt.annotate('R2 orig: %0.2f\n' % r2_S1 +
#                  #  'R2 lin: %0.2f\n' % r2_lin_S1 +
#                  'R2 ML: %0.2f\n' % r2_ML_S1 +
#                  'MAE orig: %0.2f\n' % MAE_S1 +
#                  #  'MAE lin: %0.2f\n' % MAE_lin_S1 +
#                  'MAE ML: %0.2f' % MAE_ML_S1,
#                  (8.5, 0.5),
#                  bbox=dict(facecolor='white', alpha=0.5),
#                  ha='right')
#     plt.tight_layout()

#     ax = fig.add_subplot(122)
#     plt.plot(df_all['xTB_T1'], df_all['TDDFT_T1'], '.', color=colors[0], markersize=1, label='xTB-sTDA')
#     plt.plot(df_all['pred_T1'], df_all['TDDFT_T1'], '.', color=colors[2], markersize=1, label='xTB-ML')
#     plt.plot(x, x, 'k--')
#     plt.xlim(0, 9)
#     plt.ylim(0, 9)
#     plt.gca().set_aspect('equal', adjustable='box')
#     plt.xlabel('xTB-sTDA T1 (eV)')
#     plt.ylabel('TDDFT T1 (eV)')
#     plt.legend()
#     plt.annotate('R2 orig: %0.2f\n' % r2_T1 +
#                  #  'R2 lin: %0.2f\n' % r2_lin_T1 +
#                  'R2 ML: %0.2f\n' % r2_ML_T1 +
#                  'MAE orig: %0.2f\n' % MAE_T1 +
#                  #  'MAE lin: %0.2f\n' % MAE_lin_T1 +
#                  'MAE ML: %0.2f' % MAE_ML_T1,
#                  (8.5, 0.5),
#                  bbox=dict(facecolor='white', alpha=0.5),
#                  ha='right')
#     plt.tight_layout()

#     plt.savefig('large_mol_testset_xTB_ML.png')

gen_train_data()
train_ML()
# gen_test_data()
# predict_ML()
# evaluate_predictions_3k_largemol()
# evaluate_predictions_testset()
