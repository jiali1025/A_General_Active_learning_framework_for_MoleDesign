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

'''
If the time is limited the experimentalist can use UMAP for visulization and choose the
right number of clusters according to experimentalist's understanding and the visulization.
Using dimension reduction method to visualize the number of appropriate clusters to choose.
Also it is useful after the clustering to visualize the results.
'''
data_folder_path = '/Users/lijiali/Desktop/Active leanring Framework/data/labelled_mol_data'
figure_save_folder = '/Users/lijiali/Desktop/Active leanring Framework/Figures'

df1 = pd.read_csv(data_folder_path + '/emit_8_fingerprint.csv')
df2 = pd.read_csv(data_folder_path + '/iden_emit_8.csv')
TTA_list = df2['FOM_TTA'].to_numpy()
df3 = pd.read_csv(data_folder_path + '/clustered_labelled_emitter_8_3_clusters.csv')
cluster_label = df3['label'].to_numpy()


def data_scatter(x,index_numpy, title_name):
    # choose a color palette with seaborn
    num_classes = len(np.unique(index_numpy))
    palette = np.array(sns.color_palette('hls', num_classes))
    # create a scatter plot
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect = 'equal')
    sc = ax.scatter(x[:,0], x[:, 1], lw=0, s=40, c=palette[index_numpy.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('tight')
    plt.title(title_name, fontsize=24)

    plt.savefig(figure_save_folder + '/' + title_name +'.png')

    return f, ax, sc

def data_scatter_simple(x, title_name):
    # choose a color palette with seaborn

    # create a scatter plot
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect = 'equal')
    sc = ax.scatter(x[:,0], x[:, 1], lw=0, s=40)
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('tight')
    plt.title(title_name, fontsize=24)

    plt.savefig(figure_save_folder + '/' + title_name +'.png')

    return f, ax, sc



x = df1.iloc[:,1:1025]
y = df1.iloc[:, 1025]
df_result = pd.DataFrame(y)
x = x.to_numpy()

'''
Use PCA to reduce the dimension firstly before UMAP
'''

pca = PCA(n_components=50)
pca_result = pca.fit_transform(x)

'''
Use cosine similarity as the metric according to references and align with
Kmeans. Can comment the data_scatter before the cluster_label is gotten.
'''

reducer = umap.UMAP(metric='cosine')
embedding_fingerprint_data = reducer.fit_transform(x)
data_scatter(embedding_fingerprint_data, TTA_list)
data_scatter(embedding_fingerprint_data, cluster_label)
data_scatter_simple(embedding_fingerprint_data)

