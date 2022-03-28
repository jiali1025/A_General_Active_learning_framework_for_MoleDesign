import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import numpy as np
import nltk
from nltk.cluster.kmeans import KMeansClusterer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score as sil_score, f1_score, homogeneity_score
import timeit
from rdkit.Chem import AllChem
from rdkit import Chem
data_folder_path = '/Users/lijiali/Desktop/Active leanring Framework/data/labelled_mol_data'
figure_save_folder = '/Users/lijiali/Desktop/Active leanring Framework/Figures/'

def morgan_calculation(smiles):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol,3,useChirality=True,nBits=1024)
    return np.array(fp).astype(int)


df1 = pd.read_csv(data_folder_path + '/iden_emit_8.csv')
columns = list ( range ( 1024 ) )
fp_arr = []
length = len(df1)
for i in range(length):
    smiles = df1.loc[i,"SMILES"]
    fp_arr.append(morgan_calculation(smiles))

df1['fingerprint'] =  [morgan_calculation(x) for x in df1[ "SMILES" ]]
np_fp = np.array(fp_arr)
morgan_df = pd.DataFrame ( np_fp , columns = columns )
morgan_df['smiles'] = df1["SMILES"]

df1 = morgan_df

'''
Using Kmeans clustering to unsupervisely cluster the emitter molecules with + 1 cluster to the 
visulization if time is limited otherwise can determine the optimal number of clustering by using
elbow diagram on SSE and silhouette score
As the dimenstion is quite high 1024 dimensions, it seems like fuzzy kmeans doesn't work. And
there is no available package implement methods to solve this problem.
As a result, we will just use kmeans. Also kmeans can work since the reason we cluster is to avoid
there do have clusters. We wanna pick the molecules similar to all available clusters but not choose
some averages that may result molecules not similar to any cluster. Even there is no clusters at all
this clustering process will not result in phenomenon we don't want.
The kmeans distance metrics is a choice. According to the paper Why is Tanimoto index an appropriate choice for fingerprint-based similarity calculations?
Cosine, tanimoto and so on are all suitable and rdkit get them to calculate the similarity of fp
fp is bit vector so L2 may not be suitable.
However sklearn's kmeans only has L2 to calculate the distance in loss
so I used nltk instead, it has cosine simiarlity, for other similarity there is no
package implementation so if I implement by myself can be time-consuming and errorous
Because of this we will use cosine similarity for fingerprint similarity calculation as well.
'''

x = df1.iloc[:,0:1024]
y = df1.iloc[:, 1024]
df_result = pd.DataFrame(y)
x = x.to_numpy()
y = y.to_numpy()

def run_kmeans(X, y, title):
    kclusters = list(np.arange(2, 20, 1))
    sil_scores = []
    homo_scores = []
    train_times = []

    for k in kclusters:
        start_time = timeit.default_timer()
        km = KMeansClusterer(k, distance=nltk.cluster.util.cosine_distance, repeats=10)
        assigned_clusters = km.cluster(x, assign_clusters=True)
        end_time = timeit.default_timer()
        train_times.append(end_time - start_time)
        sil_scores.append(sil_score(X, assigned_clusters))
        homo_scores.append(homogeneity_score(y, assigned_clusters))

    # elbow curve for silhouette score
    plt.plot(list(map(int, kclusters)), sil_scores)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.grid(True)
    plt.xlabel('No. Clusters')
    plt.ylabel('Avg Silhouette Score')
    plt.title('Silhouette Score Plot for KMeans: ' + title)
    plt.savefig(figure_save_folder+'Silhouette Score Plot for KMeans: ' + title +'.png')
    plt.clf()


    # plot homogeneity scores
    plt.plot(list(map(int, kclusters)), homo_scores)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.grid(True)
    plt.xlabel('No. Clusters')
    plt.ylabel('Homogeneity Score')
    plt.title('Homogeneity Scores KMeans: ' + title)
    plt.savefig(figure_save_folder+'Homogeneity Scores KMeans: ' + title + '.png')
    plt.clf()

run_kmeans(x,y, 'round8')


# kclusterer = KMeansClusterer(4, distance=nltk.cluster.util.cosine_distance, repeats=10)
#
# assigned_clusters = kclusterer.cluster(x, assign_clusters=True)
# df_result_cos = df_result.copy()
# df_result['label'] = kmeans.labels_
# df_result_cos['label'] = assigned_clusters
# df_result_cos.to_csv('./emitter_data/clustered_labelled_emitter_8_4_clusters.csv')