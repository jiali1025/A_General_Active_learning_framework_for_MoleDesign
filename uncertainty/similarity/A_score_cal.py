import pandas as pd
import numpy as np
import math
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit import DataStructs
import math
from tqdm import tqdm
from rdkit.DataStructs.cDataStructs import BulkCosineSimilarity
from collections import Counter
'''
This script will contain all the active learning acquisition score 
'''

'''
The A-score calculation for sensitizer and emitter property enhanced A-score. More like an
optimization approach. To find the molecules with desired properties and assign them scores.
'''

'''
The current version is not optimized as the prediction dataset is split into different
portions. 
'''

'''
The difference for sensitizers and emitters are the difference in the target property
'''

def simple_sensitizer_A_score(E_T1, E_S1, unc_S1, unc_T1, decay_para, target_weight, uncertainty_weight):
    '''

    :param E_T1: A numpy list contain the T1 energy value
    :param E_S1: A numpy list contain the S1 energy value
    :param unc_S1: A numpy list contain the S1 uncertainty value
    :param unc_T1: A numpy list contain the T1 uncertainty value
    :param decay_para: A hyperparameter to control the decay of target property term bigger the steeper the decay
    :param target_weight: A hyperparameter to control the target term's portion in the acquisition score calculation
    :param uncertainty_weight: A hyperparameter to control the uncertainty term's portion in the acquisition score calculation
    :return: The calculated A score
    '''
    return target_weight * np.exp(-decay_para*np.abs(1-E_T1/E_S1)) + uncertainty_weight * (unc_S1 + unc_T1)

def simple_emitter_A_score(E_T1, E_S1, unc_S1, unc_T1, decay_para, target_weight, uncertainty_weight):
    '''

    :param E_T1: A numpy list contain the T1 energy value
    :param E_S1: A numpy list contain the S1 energy value
    :param unc_S1: A numpy list contain the S1 uncertainty value
    :param unc_T1: A numpy list contain the T1 uncertainty value
    :param decay_para: A hyperparameter to control the decay of target property term bigger the steeper the decay
    :param target_weight: A hyperparameter to control the target term's portion in the acquisition score calculation
    :param uncertainty_weight: A hyperparameter to control the uncertainty term's portion in the acquisition score calculation
    :return: The calculated A score
    '''
    return target_weight * np.exp(-decay_para*np.abs(2-E_S1/E_T1)) + uncertainty_weight * (unc_S1 + unc_T1)


'''
Molecular similarity based A score which add in domain knowledge term of molecular similarity into
the A score calculation. Since the embedding deep neural network will change for different rounds of
active learning, a more generalized and stable form of embedding by using morgan fingerprint is implemented.
'''

def Average(list):
    return sum(list) / len(list)

def morgan_fp_calculation(smiles):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol,3,useChirality=True,nBits=1024)
    return fp

def molecular_similarity_calculation_parallel(smiles, cluster_labelled_molecule_fp_list):
    '''
    It is an accelerated version of molecular similarity calculation function using the parallel
    computing

    :param smiles: A query smile of molecules to screen over, for example a molecule belong to the whole
    design space of molecules' smiles.
    :param cluster_labelled_molecule_fp_list: It is in the form of a list of list[[],[],...]
    for each list in this list is the fingerprint of molecules in one cluster. The number of clusters
    is determined by the kmeans experiment. And this list can be either from the labelled emitter data
    or the labelled sensitizer data.
    :param labelled_data: The original labelled list of either emitters or sensitizers to ensure the function is
    implemented correctly
    :return: The similarity score calculated of the query smile to the cluster of smiles
    that the query smiles have the highest average similarity to (i.e., the query smile has higher probability
    belong to this cluster)
    '''
    mol = Chem.MolFromSmiles(smiles)
    fp_to_compare = AllChem.GetMorganFingerprintAsBitVect(mol,3,useChirality=True,nBits=1024)
    number_of_cluster = len(cluster_labelled_molecule_fp_list)
    Total_length = 0
    for i in range(number_of_cluster):
        Total_length += len(cluster_labelled_molecule_fp_list[i])
    # Total_length_of_label = len(labelled_data)
    # assert Total_length == Total_length_of_label
    combine_list_of_score_cluster = []
    combine_list_of_average_score_cluster = []


    for n in range(number_of_cluster):
        cluster_score_array = BulkCosineSimilarity(fp_to_compare, cluster_labelled_molecule_fp_list[n])
        average_score_in_one_cluster = Average(cluster_score_array)
        combine_list_of_average_score_cluster.append(average_score_in_one_cluster)
        combine_list_of_score_cluster.append(sorted(cluster_score_array, reverse=True))

    Max_avg_similarity = max(combine_list_of_average_score_cluster)
    max_cluster_index = combine_list_of_average_score_cluster.index(Max_avg_similarity)
    Max_similarity_score = Average(combine_list_of_score_cluster[max_cluster_index][:100])


    return Max_similarity_score


def similarity_A_score(df_labelled_molecule, df_screen, E_T1, E_S1, unc_S1, unc_T1, decay_para, target_weight,
                       uncertainty_weight, domain_weight, is_sensitizer):
    '''

    :param df_labelled_molecule: It is the labelled molecule dataset (in the df form) with cluster labels start from 0
    :param df_screen: It is a df contains the design space to screen over
    :param E_T1: A numpy list contain the T1 energy value
    :param E_S1: A numpy list contain the S1 energy value
    :param unc_S1: A numpy list contain the S1 uncertainty value
    :param unc_T1: A numpy list contain the T1 uncertainty value
    :param decay_para: A hyperparameter to control the decay of target property term bigger the steeper the decay
    :param target_weight: A hyperparameter to control the target term's portion in the acquisition score calculation
    :param uncertainty_weight: A hyperparameter to control the uncertainty term's portion in the acquisition score calculation
    :param domain_weight: A hyperparameter to control the domain specific term's portion in the acquisition score calculation
    :param is_sensitizer: A boolen to determine calculate for sensitizer A score or emitter A score
    :return: The calculated A score
    '''

    labelled_molecule_fp_list = []
    length_of_labelled_molecules = len(df_labelled_molecule)
    molecule_fp_cluster_label = []
    for i in tqdm(range(length_of_labelled_molecules)):
        smiles = df_labelled_molecule.loc[i, "SMILES"]
        labelled_molecule_fp_list.append(morgan_fp_calculation(smiles))
        molecule_fp_cluster_label.append(df_labelled_molecule.loc[i, "label"])
    labelled_fp_list_length = len(labelled_molecule_fp_list)
    number_of_clusters = len(Counter(molecule_fp_cluster_label).keys())
    cluster_labelled_molecule_fp_list = []
    for n in range(number_of_clusters):
        cluster_labelled_molecule_fp_list.append([])

    for i in tqdm(range(labelled_fp_list_length)):
        for n in range(number_of_clusters):
            if molecule_fp_cluster_label[i] == n:
                cluster_labelled_molecule_fp_list[n].append(labelled_molecule_fp_list[i])

    df_screen = df_screen
    length_screen = len(df_screen)
    mol_similarity_score = []
    for i in tqdm(range(length_screen)):
        smiles = df_screen.loc[i, 'smiles']
        similarity_score = molecular_similarity_calculation_parallel(smiles, cluster_labelled_molecule_fp_list,
                                                                     labelled_molecule_fp_list)

        mol_similarity_score.append(similarity_score)

    mol_similarity_score_numpy_array = np.array(mol_similarity_score)
    if is_sensitizer:
        return target_weight * np.exp(-decay_para*np.abs(1-E_T1/E_S1)) + uncertainty_weight * (unc_S1 + unc_T1) + domain_weight * (mol_similarity_score_numpy_array)
    else:
        return target_weight * np.exp(-decay_para*np.abs(2-E_S1/E_T1)) + uncertainty_weight * (unc_S1 + unc_T1) + domain_weight * (mol_similarity_score_numpy_array)


