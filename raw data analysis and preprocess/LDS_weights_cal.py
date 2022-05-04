import os
import shutil
import torch
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang
from scipy.ndimage import convolve1d
from torch.utils import data
import pandas as pd
import matplotlib.pyplot as plt
def get_lds_kernel_window(kernel, ks, sigma):
    assert kernel in ['gaussian', 'triang', 'laplace']
    half_ks = (ks - 1) // 2
    if kernel == 'gaussian':
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
    elif kernel == 'triang':
        kernel_window = triang(ks)
    else:
        laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
        kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(map(laplace, np.arange(-half_ks, half_ks + 1)))

    return kernel_window


def prepare_weights( data, reweight, max_target=51, lds=False, lds_kernel='gaussian', lds_ks=5, lds_sigma=2):
    assert reweight in {'none', 'inverse', 'sqrt_inv'}
    assert reweight != 'none' if lds else True, \
        "Set reweight to \'sqrt_inv\' (default) or \'inverse\' when using LDS"

    value_dict = {x: 0 for x in range(max_target)}
    data = data.to_numpy()
    labels = data[:, -1].tolist()
    # mbr
    for label in labels:
        value_dict[min(max_target - 1, int(label))] += 1
    if reweight == 'sqrt_inv':
        value_dict = {k: np.sqrt(v) for k, v in value_dict.items()}
    elif reweight == 'inverse':
        value_dict = {k: np.clip(v, 5, 20000) for k, v in value_dict.items()}  # clip weights for inverse re-weight
    num_per_label = [value_dict[min(max_target - 1, int(label))] for label in labels]
    if not len(num_per_label) or reweight == 'none':
        return None
    print(f"Using re-weighting: [{reweight.upper()}]")

    if lds:
        lds_kernel_window = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)
        print(f'Using LDS: [{lds_kernel.upper()}] ({lds_ks}/{lds_sigma})')
        smoothed_value = convolve1d(
            np.asarray([v for _, v in value_dict.items()]), weights=lds_kernel_window, mode='constant')
        num_per_label = [smoothed_value[min(max_target - 1, int(label))] for label in labels]

    weights = [np.float32(1 / x) for x in num_per_label]
    scaling = len(weights) / np.sum(weights)
    weights = [scaling * x for x in weights]
    return weights

if __name__ == '__main__':
    figure_save_folder = '/home/pengfei/projects/AL_framework/Figures'

    df1 = pd.read_csv('./preprocessed_data/whole_design_space_ratio.csv')
    # too_big_outliers = df1['T1/S1'].nlargest(n=3).index
    '''
    remove too big outliers
    '''
    df_without_big = df1[df1['T1/S1'] <=1.2]
    df_filtered = df_without_big[df_without_big['T1/S1'] >= 0]


    df_filtered['T1/S1'] = df_filtered['T1/S1'] * 100
    weights = prepare_weights(df_filtered, reweight='sqrt_inv', max_target=120, lds=True, lds_kernel='gaussian', lds_ks=5, lds_sigma=2)
    df_filtered['weights'] = weights
    data_weights_df = pd.DataFrame(weights)
    df_filtered.to_csv('./preprocessed_data/whole_train_LDS_trail1.csv')
    data_weights_df.to_csv('./preprocessed_data/whole_train_LDS_weights.csv')
    '''
    plot
    '''
    plt.hist(x=weights, bins=200,
             color="steelblue",
             edgecolor="black")
    # plt.ylim((0,1000))
    # plt.xlim((0, 1.2))
    plt.xlabel("weights")
    plt.ylabel("frequency")
    plt.title("weights_for_training")
    plt.savefig(figure_save_folder + "/weights_for_training_LDS_inverse.png")

    plt.clf()
    # plt.ylim((0,1000))
    plt.hist(x=df_filtered['T1/S1'], bins=200,
             color="steelblue",
             edgecolor="black")
    # plt.xlim((0, 1.2))
    plt.xlabel("ratio")
    plt.ylabel("frequency")
    plt.title("ratio")
    plt.savefig(figure_save_folder + "/whole_data_visulization.png")
    plt.clf()
    plt.scatter(weights,df_filtered['T1/S1'])
    plt.show()
