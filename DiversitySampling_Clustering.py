import os
from random import shuffle

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from Cluster import CosineClusters
from Cluster import Cluster
from PreProcess import PCA_scale

wrk_path_3 = r"/Users/calvin/Documents/OneDrive/Documents/2020/Liposomes Vitamins/LiposomeFormulation"
os.chdir(wrk_path_3)

'''
Pull in Unlabelled data
'''


# TODO need to determine how to implement it into a class (is it even required?)

def unlabelled_data(file, method):
    ul_df = pd.read_csv(file)
    column_drop = ['Duplicate_Check',
                   'PdI Width (d.nm)',
                   'PdI',
                   'Z-Average (d.nm)',
                   'ES_Aggregation']
    ul_df = ul_df.drop(columns=column_drop)
    ul_df.replace(np.nan, 'None', inplace=True)
    ul_df = pd.get_dummies(ul_df, columns=["Component_1", "Component_2", "Component_3"],
                           prefix="", prefix_sep="")
    # if method=='fillna':    ul_df['Component_3'] = ul_df['Component_3'].apply(lambda x: None if pd.isnull(x) else x) #TODO This should be transformed into an IF function, thus when the function for unlabelled is filled with a parameter, then activates

    ul_df = ul_df.groupby(level=0, axis=1, sort=False).sum()

    print(ul_df.isna().any())
    X_val = ul_df.to_numpy()
    columns_x_val = ul_df.columns
    return X_val, columns_x_val


X_val, columns_x_val = unlabelled_data('unlabelled_data_full.csv',
                                       method='fillna')  # TODO need to rename


class DiversitySampling():

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.pca = PCA_scale()

    def get_cluster_samples(self, data, num_clusters=10, max_epochs=10, limit=-1):

        if limit > 0:
            data_new = list(zip(data[0], data[1]))
            shuffle(data_new)

            a, b = zip(*data_new)
            a = np.array(a)
            b = np.array(b)
            data = tuple([a, b])
            #shuffle(data)
            data = data[:limit]

        self.cosine_clusters = CosineClusters(num_clusters)

        self.cosine_clusters.add_random_training_items(data[0], data[1])

        for i in range(0, max_epochs):
            print('Epoch ' + str(i))
            added = self.cosine_clusters.add_items_to_best_cluster(data[0], data[1])
            if added == 0:
                break

        self.centroids = self.cosine_clusters.get_centroids()
        self.outliers = self.cosine_clusters.get_outliers(2) #Changed it to 2...
        self.randoms = self.cosine_clusters.get_randoms(3)

        return self.centroids, self.outliers, self.randoms

    def get_representative_samples(self, training_data, unlabeled_data, number=20, limit=10000):

        if limit > 0:
            shuffle(training_data)
            training_data = training_data[:limit]

            shuffle(unlabeled_data)
            unlabeled_data = unlabeled_data[:limit]

        training_cluster = Cluster()
        for index, item in enumerate(training_data):
            training_cluster.add_to_cluster(item)

        unlabeled_cluster = Cluster()
        for index, item in enumerate(unlabeled_data):
            unlabeled_cluster.add_to_cluster(item)

        for index, item in enumerate(unlabeled_data):
            training_score = training_cluster.cosine_similarity(item)
            unlabeled_score = unlabeled_cluster.cosine_similarity(item)

            representativeness = unlabeled_score - training_score

        unlabeled_data.sort(reverse=True, key=lambda x: x[4])
        return unlabeled_data[:number:]

    def get_adaptive_representative_samples(self, training_data, unlabeled_data, number=20, limit=5000):

        samples = []
        for i in range(0, number):
            print("Epoch " + str(i))
            representative_item = self.get_representative_samples(training_data, unlabeled_data, 1, limit)[0]
            samples.append(representative_item)
            unlabeled_data.remove(representative_item)

        return samples

    def graph_clusters(self, data):

        sample_y = [self.cosine_clusters.clusters.index(_) for _ in self.cosine_clusters.item_cluster.values()]

        for index, cluster in enumerate(self.cosine_clusters.clusters):
            sample_sort = cluster.distance_sort()

        # Extract out the centroids/outliers/randoms
        centroid_array = []
        for _ in self.centroids:
            for i in _:
                centroid_array.append(i[2])

        outlier_array = []
        for _ in self.outliers:
            for i in _:
                outlier_array.append(i[2])

        random_array = []
        for _ in self.randoms:
            for i in _:
                random_array.append(i[2])

        # PCA Fit & Transform Data - must be done on the fullscope of the data
        self.pca.pca_fit(data[1])
        data = self.pca.pca_transform(data[1])

        # PCA Transform
        centroid_array = self.pca.pca_transform(centroid_array)
        outlier_array = self.pca.pca_transform(outlier_array)
        random_array = self.pca.pca_transform(random_array)



        D_id_color = [u'orchid', u'darkcyan', u'dodgerblue', u'turquoise', u'darkviolet', u'chartreuse', u'gold',
                      u'tomato', u'crimson', u'yellow', u'maroon', u'black', u'bisque', u'aqua', u'navy', u'magenta',
                      u'fuchsia', u'peru', u'red', u'brown', u'cornflowerblue']

        plt.figure(figsize=(18, 6))
        plt.subplot(131)
        plt.scatter(data[:, 0], data[:, 1])


        plt.subplot(132)
        for label in [*range(len(self.cosine_clusters.clusters))]:
            indices = [i for i, l in enumerate(sample_y) if l == label]
            current_tx = np.take(data[:, 0], indices)
            current_ty = np.take(data[:, 1], indices)
            print(label)
            color = D_id_color[label]
            print(current_tx.shape)
            plt.scatter(current_tx, current_ty, c=color, label=label)
        plt.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0)

        plt.subplot(133)
        plt.scatter(data[:, 0], data[:, 1], alpha=0.2, color='gray')
        f2 = lambda x: [_[2] for _ in x]
        for label in [*range(len(self.cosine_clusters.clusters))]:
            color = D_id_color[label]

            plt.scatter(np.take(centroid_array[label], 0), np.take(centroid_array[label], 1), c=color
                        , label=f'{label} centroids')
            plt.scatter(np.take(outlier_array[label], 0), np.take(outlier_array[label], 1), marker ='*', c=color
                        , label=f'{label} outliers')
            plt.scatter(np.take(random_array[label], 0), np.take(random_array[label], 1),marker='^', c=color
                        , label=f'{label} randoms')
            #plt.scatter(np.array(f2(self.centroids[label]))[:, 0], np.array(f2(self.centroids[label]))[:, 1], c=color,
               #         label=f'{label} centroids')
            #plt.scatter(np.array(f2(self.outliers[label]))[:, 0], np.array(f2(self.outliers[label]))[:, 1], marker='*',
             #           c=color,
             #           label=f'{label} outliers')
            #plt.scatter(np.array(f2(self.randoms[label]))[:, 0], np.array(f2(self.randoms[label]))[:, 1], marker='^',
              #          c=color,
              #          label=f'{label} randoms')

        plt.legend(bbox_to_anchor=(1.04, 0), loc="lower left", borderaxespad=0)
        plt.show()
