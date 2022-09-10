


import pandas as pd
import hdbscan
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from sklearn.preprocessing import MinMaxScaler
##Import the unlabeled data
from joblib import Memory

unlabeled_data = pd.read_csv(r'/Users/calvin/Library/CloudStorage/OneDrive-Personal/Documents/2022/RegressorCommittee_Output/AL_Random_Output/7R9HYC55ZB_20220825_2359_complete_iteration_0/Unlabeled_Data.csv')
unlabeled_data.drop(columns=['Unnamed: 0'], inplace=True)
unlabeled_data_no_idx = unlabeled_data.columns.difference(['original_index'])
#sample_data = unlabeled_data.sample(n=10000, random_state=42)
train_data = pd.read_csv(r"/Users/calvin/Library/CloudStorage/OneDrive-Personal/Documents/2022/Data_Output/Results_Complete.csv")
train_data = train_data[train_data['Z-Average (d.nm)'] <= float(1000)]
train_data = train_data[unlabeled_data_no_idx]
unlabeled_data_mod = unlabeled_data_no_idx.difference(["Lipid_Vol_Pcnt"])
unlabeled_data[unlabeled_data_mod] = np.round(unlabeled_data[unlabeled_data_mod], 2)
added_data = pd.read_csv(r"/Users/calvin/Library/CloudStorage/OneDrive-Personal/Documents/2022/RegressorCommittee_Output/AL_Output/CKVMGXBR2R_20220904_0027_complete_iteration_10/Added_Data.csv")
added_data = added_data[unlabeled_data_no_idx]
added_data[unlabeled_data_mod] = np.round(added_data[unlabeled_data_mod], 2)
merged_added_data_idx = added_data.merge(unlabeled_data, on=list(unlabeled_data_no_idx))
list_added_data_idx = merged_added_data_idx['original_index'].astype(int).to_list()
#remove data points
unlabeled_data['original_index'] = unlabeled_data['original_index'].astype(int)
unlabeled_data = unlabeled_data[~unlabeled_data.original_index.isin(list_added_data_idx)]
unlabeled_data.drop(columns=['original_index'], inplace=True)
sample_data = unlabeled_data.sample(n=10000, random_state=42)

comb_df = pd.concat([sample_data, train_data,added_data])

tsne_init = TSNE(learning_rate='auto', n_iter=5000,n_jobs=-1, random_state=42)

projection = tsne_init.fit_transform(comb_df)

plt.close("all")
plt.grid(True, alpha=0.5)
plt.scatter(*projection[:10000].T, s=15, linewidths=0.1,edgecolors='black', c='#9fb0ce',label='unlabelled_data')
plt.scatter(*projection[10000:int(train_data.shape[0])+10000].T, s=30, linewidths=0.3,edgecolors='black', c='#AC5C79', label='training_data')
plt.title("2-D Clustering of Training Data & Unlabelled Data",pad=20,loc='center', wrap=True)
plt.tick_params(axis='both', labelsize=8)
plt.legend()
plt.tight_layout()
plt.savefig(r"/Users/calvin/Library/CloudStorage/OneDrive-Personal/plot.png", dpi=400)
plt.show()



plt.close("all")
plt.rcParams['figure.edgecolor']='grey'
plt.grid(True, alpha=0.5)
plt.style.context('ggplot')
plt.scatter(*projection[:10000].T,s=15, linewidths=0.1,edgecolors='black', c='#9fb0ce',label='unlabelled_data')
plt.scatter(*projection[10000:int(train_data.shape[0])+10000].T, s=30, linewidths=0.3, edgecolors='black',c='#AC5C79', label='training_data')
plt.scatter(*projection[10000 + train_data.shape[0]:-50].T, s=30, linewidths=0.3, edgecolors='black',c='#79AC5C', label='added_data: iter 0-5')
plt.title("2-D Cluster of Training Data & Unlabelled Data & Active Learn. Added Data", pad=20,loc='center', wrap=True)
plt.tick_params(axis='both', labelsize=8)
plt.legend()
plt.tight_layout()
plt.savefig(r"/Users/calvin/Library/CloudStorage/OneDrive-Personal/plot_added_half.png", dpi=400)
plt.show()

# plt.close("all")
# plt.rcParams['figure.edgecolor']='grey'
# plt.grid(True, alpha=0.5)
# plt.style.context('ggplot')
# plt.scatter(*projection[:10000].T,s=15, linewidths=0.1, c='#9fb0ce',label='unlabelled_data')
# plt.scatter(*projection[10000:int(train_data.shape[0])+10000].T, s=30, linewidths=0.3, c='#AC5C79', label='training_data')
# plt.scatter(*projection[10000 + train_data.shape[0]:].T, s=30, linewidths=0.3, c='#79AC5C', label='added_data')
# plt.title("2-D Cluster of Training Data & Unlabelled Data & Added Data", pad=20,loc='center', wrap=True)
# plt.tick_params(axis='both', labelsize=8)
# plt.legend()
# plt.tight_layout()
#
# plt.savefig(r"/Users/calvin/Library/CloudStorage/OneDrive-Personal/plot_added_full.png", dpi=400)
# plt.show()
#sns.distplot(clusterer.outlier_scores_[np.isfinite(clusterer.outlier_scores_)], rug=True)
#plt.show()



plt.close("all")
plt.rcParams['figure.edgecolor']='grey'
plt.grid(True, alpha=0.5)
plt.style.context('ggplot')
plt.scatter(*projection[:10000].T,s=15, linewidths=0.1,edgecolors='black', c='#9fb0ce',label='unlabelled_data')
plt.scatter(*projection[10000:int(train_data.shape[0])+10000].T, s=30, linewidths=0.3, edgecolors='black',c='#AC5C79', label='training_data')
plt.scatter(*projection[10000 + train_data.shape[0]:-50].T, s=30, linewidths=0.3,edgecolors='black', c='#79AC5C', label='added_data: iter 0-5')
plt.scatter(*projection[10000 + train_data.shape[0]+50:].T, s=30, linewidths=0.3,edgecolors='black', c='orange', label='added_data: iter 6-10', alpha=0.7)
plt.title("2-D Cluster of Training Data & Unlabelled Data & Active Learn. Added Data", pad=20,loc='center', wrap=True)
plt.tick_params(axis='both', labelsize=8)
plt.legend()
plt.tight_layout()
plt.savefig(r"/Users/calvin/Library/CloudStorage/OneDrive-Personal/plot_added_full_split.png", dpi=400)
plt.show()

print('Why')

pce = PCA(n_components=2)
pce.fit(comb_df)
ul_data = pce.transform(sample_data)
tr_data = pce.transform(train_data)

plt.close("all")
plt.scatter(*ul_data.T, s=50, linewidths=0, c='b')
plt.scatter(*tr_data.T, s=50, linewidths=0, c='r')
plt.show()
print('next')


#####
added_data = pd.read_csv(r"/Users/calvin/Library/CloudStorage/OneDrive-Personal/Documents/2022/RegressorCommittee_Output/Random_Output/HT3WB7JTCL_20220903_1803_complete_iteration_10/Added_Data.csv")
added_data = added_data[unlabeled_data_no_idx]
added_data[unlabeled_data_mod] = np.round(added_data[unlabeled_data_mod], 2)
merged_added_data_idx = added_data.merge(unlabeled_data, on=list(unlabeled_data_no_idx))
#list_added_data_idx = merged_added_data_idx['original_index'].astype(int).to_list()
comb_df = pd.concat([sample_data, train_data,added_data])

tsne_init = TSNE(learning_rate='auto', n_iter=5000,n_jobs=-1, random_state=42)
projection = tsne_init.fit_transform(comb_df)
plt.close("all")
plt.rcParams['figure.edgecolor']='grey'
plt.grid(True, alpha=0.5)
plt.style.context('ggplot')
plt.scatter(*projection[:10000].T,s=15, linewidths=0.1,edgecolors='black', c='#9fb0ce',label='unlabelled_data')
plt.scatter(*projection[10000:int(train_data.shape[0])+10000].T, s=30, linewidths=0.3, edgecolors='black',c='#AC5C79', label='training_data')
plt.scatter(*projection[10000 + train_data.shape[0]:-50].T, s=30, linewidths=0.3, edgecolors='black',c='#79AC5C', label='added_data: iter 0-5')
plt.title("2-D Cluster of Training Data & Unlabelled Data & Random Learn. Added Data", pad=20,loc='center', wrap=True)
plt.tick_params(axis='both', labelsize=8)
plt.legend()
plt.tight_layout()
plt.savefig(r"/Users/calvin/Library/CloudStorage/OneDrive-Personal/plot_added_half_random.png", dpi=400)
plt.show()

plt.close("all")
plt.rcParams['figure.edgecolor']='grey'
plt.grid(True, alpha=0.5)
plt.style.context('ggplot')
plt.scatter(*projection[:10000].T,s=15, linewidths=0.1,edgecolors='black', c='#9fb0ce',label='unlabelled_data')
plt.scatter(*projection[10000:int(train_data.shape[0])+10000].T, s=30, linewidths=0.3, edgecolors='black',c='#AC5C79', label='training_data')
plt.scatter(*projection[10000 + train_data.shape[0]:-50].T, s=30, linewidths=0.3,edgecolors='black', c='#79AC5C', label='added_data: iter 0-5')
plt.scatter(*projection[10000 + train_data.shape[0]+50:].T, s=30, linewidths=0.3,edgecolors='black', c='orange', label='added_data: iter 6-10', alpha=0.7)
plt.title("2-D Cluster of Training Data & Unlabelled Data & Random Learn. Added Data", pad=20,loc='center', wrap=True)
plt.tick_params(axis='both', labelsize=8)
plt.legend()
plt.tight_layout()
plt.savefig(r"/Users/calvin/Library/CloudStorage/OneDrive-Personal/plot_added_full_split_random.png", dpi=400)
plt.show()