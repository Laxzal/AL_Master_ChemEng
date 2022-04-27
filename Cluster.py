from typing import List

import hdbscan
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.style as style
import numpy as np
import pandas as pd

from random import shuffle

from joblib import Parallel, delayed, Memory
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples


class CosineClusters():

    def __init__(self, num_clusters: int = 100, Euclidean=False):

        self.clusters = []
        self.item_cluster = {}
        self.Euclidean = Euclidean
        # Create Initial Cluster
        for i in range(0, num_clusters):
            self.clusters.append(Cluster())

    def add_random_training_items(self, index_unlabelled, unlabelled):
        cur_index = 0
        for index, item in zip(index_unlabelled, unlabelled):
            self.clusters[cur_index].add_to_cluster(index, item)
            formulation_id = index
            self.item_cluster[formulation_id] = self.clusters[cur_index]

            cur_index += 1
            if cur_index >= len(self.clusters):
                cur_index = 0

    def add_items_to_best_cluster(self, index_unlabelled, unlabelled):
        added = 0
        for index, item in zip(index_unlabelled, unlabelled):
            print(added)
            new = self.add_item_to_best_cluster(index, item)
            if new:
                added += 1

        return added

    def add_item_to_best_cluster(self, index, item):

        best_cluster = None
        best_fit = float("-inf")
        previous_cluster = None

        formulation_id = index
        if formulation_id in self.item_cluster:
            previous_cluster = self.item_cluster[formulation_id]
            previous_cluster.remove_from_cluster(formulation_id, item)

        for cluster in self.clusters:
            fit = cluster.cosine_similarity(item, Euclidean=self.Euclidean)

            if fit > best_fit:
                best_fit = fit
                best_cluster = cluster

        best_cluster.add_to_cluster(index, item)
        self.item_cluster[formulation_id] = best_cluster

        if best_cluster == previous_cluster:
            return False
        else:
            return True

    def get_centroids(self, number_per_cluster=1):
        centroids = []
        for cluster in self.clusters:
            centroids.append(cluster.get_centroid(number_per_cluster))

        return centroids

    def get_outliers(self, number_per_cluster=1):
        outliers = []
        for cluster in self.clusters:
            outliers.append(cluster.get_outlier(number_per_cluster))

        return outliers

    def get_randoms(self, number_per_cluster=1):
        randoms = []
        for cluster in self.clusters:
            randoms.append(cluster.get_random_members(number_per_cluster))

        return randoms


class Cluster():
    feature_idx = {}

    def __init__(self, Euclidean=False):
        self.members = {}
        self.feature_vector = None
        self.Euclidean = Euclidean
        self.distance = []

    def add_to_cluster(self, index, item):

        formulation_id = index
        data = item
        self.members[formulation_id] = item

        try:
            if self.feature_vector == None:
                self.feature_vector = data
        except:
            self.feature_vector = self.feature_vector + data

        # for feature in features:
        #    while len(self.feature_vector) <= feature:
        #        self.feature_vector.append(0)

    #            self.feature_vector[feature] += 1

    def remove_from_cluster(self, index, item):
        formulation_id = index
        data = item

        exists = self.members.pop(formulation_id, False)

        if exists is not None:
            self.feature_vector = self.feature_vector - data

    def cosine_similarity(self, item, Euclidean=False):
        data = item
        center_vec = self.feature_vector / len(list(self.members.keys()))

        #item_tensor = torch.FloatTensor(data)
        #center_tensor = torch.FloatTensor(center_vec)

        if Euclidean:
            similarity = -np.sqrt(np.sum(np.square(data - center_vec)))
            return similarity
        else:
            similarity = F.cosine_similarity(item_tensor, center_tensor, 0)
            return similarity.item()  # converts to float

    def size(self):
        return len(self.members.keys())

    def distance_sort(self):
        self.distance = []
        for formulation_id in self.members.keys():
            item = self.members[formulation_id]
            similarity = self.cosine_similarity(item, Euclidean=self.Euclidean)
            # self.distance.append([similarity, item[0], item[1]])
            self.distance.append([similarity, formulation_id, item])
        self.distance.sort(reverse=True, key=lambda x: x[0])
        return self.distance

    def get_centroid(self, number=1):
        if len(self.members) == 0:
            return []
        return self.distance_sort()[:number]

    def get_outlier(self, number=1):
        if len(self.members) == 0:
            return {}
        return self.distance_sort()[-number:]

    def get_random_members(self, number=1):
        if len(self.members) == 0:
            return []
        _ = self.distance_sort()
        randoms = []
        for i in range(0, number):
            randoms.append(_[np.random.randint(len(self.members))])

        return randoms


class KMeans_Cluster():

    def __init__(self, unlabeled_data: np.ndarray, n_clusters: int = 5, n_init: str = 'k-means++',
                 max_iteration: int = 500,
                 algorithm: str = 'auto'):
        self.kmeans = KMeans(n_clusters=n_clusters, init=n_init, max_iter=max_iteration, algorithm=algorithm,
                             random_state=42)
        self.unlabeled_data_index = unlabeled_data[0]
        self.unlabeled_data = unlabeled_data[1]

        self.n_init = n_init
        self.algorithm = algorithm

    def kmeans_fit(self):
        self.kmeans.fit(self.unlabeled_data)

    def kmeans_intertia(self):
        self.kmeans.inertia_

        return self.kmeans.inertia_

    def elbow_method(self, clusters: int = 5):
        SSE = []
        for cluster in range(1, clusters):
            kmeans = KMeans(n_clusters=cluster, init=self.n_init, algorithm=self.algorithm)
            kmeans.fit(self.unlabeled_data)
            SSE.append(kmeans.inertia_)

        frame = pd.DataFrame({'Cluster': range(1, clusters), 'SSE': SSE})
        plt.figure(figsize=(12, 6))
        plt.plot(frame['Cluster'], frame['SSE'], marker='o')
        plt.xlabel('Number of clusters')
        plt.ylabel('Inertia')
        plt.show()

    def kMeansRes(self, scaled_data, k: int, alpha: float = 0.01):
        '''

        :param scaled_data: matrix - Scaled data rows are samples and columns are the features for clustering
        :param k: int - current k for applying kmeans
        :param alpha: float - manually turned factor that gives a penality to the number of clusters
        :return scaled inertia:
        '''

        interia_o = np.square((scaled_data - scaled_data.mean(axis=0))).sum()
        # git k-means
        kmeans = KMeans(n_clusters=k, init=self.n_init, algorithm=self.algorithm, random_state=0).fit(scaled_data)
        scaled_inertia = (kmeans.inertia_ / interia_o) + (alpha * k)
        return scaled_inertia

    def chooseBestKforKmeansParallel(self, k_range, alpha: float = 0.01):
        print('Finding Best K for KMeans...')
        ans = Parallel(n_jobs=-1, verbose=10)(
            delayed(self.kMeansRes)(self.unlabeled_data, k, alpha) for k in range(1, k_range))
        ans = list(zip(range(1, k_range), ans))
        results = pd.DataFrame(ans, columns=['k', 'Scaled Inertia']).set_index('k')
        best_k = results.idxmin()[0]
        print('Best K for Clustering: ', best_k)
        return best_k, results

    def kmeans_transform(self, data):
        transformed_array = self.kmeans.transform(data)

        return transformed_array

    def kmeans_predict(self, data):
        predict_array = self.kmeans.predict(data)

        return predict_array

    def kmeans_labels(self):
        labels = self.kmeans.labels_

        return labels

    def kmeans_centres(self):
        centres = self.kmeans.cluster_centers_
        return centres

    def create_array(self, percentile: float = 95.0, threshold: float = 1.0, n_instances: int = 100,
                     dist_measuring: str = 'euclidean'):
        x_val = self.unlabeled_data.copy()

        clusters = self.kmeans_labels()
        centroids = self.kmeans_centres()

        points = np.empty((0, len(x_val[0])), float)

        distances = np.empty((0, len(x_val[0])), float)

        for i, center_elem in enumerate(centroids):
            # CDIST is used to calculate the distance between centre and other points
            distances = np.append(distances, cdist([center_elem], x_val[clusters == 1], 'euclidean'))

            points = np.append(points, x_val[clusters == i], axis=0)

        distance_df = pd.DataFrame(distances)
        x_val = pd.DataFrame(x_val)

        x_val['distances'] = distance_df
        x_val['original_index'] = self.unlabeled_data_index
        x_val['label_cluster'] = clusters
        # x_val[f'{percentile}th_percentile'] = np.percentile(distances,percentile)
        distribution_instances = round(n_instances / len(set(clusters)))
        distance_points = {}

        for i in list(set(clusters)):
            print('Cluster: ', i)
            temp_df = x_val[x_val['label_cluster'].isin([i])]
            # distance_points = np.empty((0, len(temp_df[0])), float)
            points = np.empty((0, temp_df.shape[1] - 3), float)

            for index, value in temp_df.iterrows():
                if points.shape[0] <= distribution_instances:
                    convert_series = value.to_frame().T
                    convert_series['original_index'] = convert_series['original_index'].astype(int)
                    convert_series['label_cluster'] = convert_series['label_cluster'].astype(int)
                    formulation_id = convert_series['original_index'].values[0]
                    data = convert_series.drop(columns=['distances', 'original_index', 'label_cluster'])
                    index = index

                    if points.shape[0] >= 1:
                        distance = cdist(points[-1:, :], data, dist_measuring)
                        if distance >= threshold:
                            distance_points[formulation_id] = distance[0][0]
                            points = np.append(points, data, axis=0)
                        # distance_points = np.append(distance_points, cdist(points[-1],data,'euclidean'))
                    else:
                        points = np.append(points, data, axis=0)

        print('Completed distance measuring...')

        distances_df = pd.DataFrame.from_dict(distance_points, orient='index')
        distances_df = distances_df.rename(columns={0: 'distances_local'})
        result = pd.merge(x_val, distances_df, left_on='original_index', right_index=True)
        results_index = result['original_index']
        distance_score = result['distances_local']
        result.drop(columns=['distances', 'original_index', 'label_cluster', 'distances_local'], inplace=True)

        return result, results_index, distance_score

    def silhouette(self, X: np.ndarray, range_clusters: List[int] = [2, 3, 4, 5, 6, 7, 8, 9]):
        silhouette_avg_n_clusters = []

        for n_clusters in range_clusters:


            # Initiliaze Clusterer with n_clusters value and a random generator seed of 10 for reproducibility
            clusterer = KMeans(n_clusters=n_clusters, init=self.n_init, algorithm=self.algorithm, random_state=42)
            cluster_labels = clusterer.fit_predict(X)

            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed
            # clusters
            silhouette_avg = silhouette_score(X, cluster_labels, n_jobs = -1)
            print("For n_clusters = ", n_clusters,
                  "The average silhouette score is: ", silhouette_avg)

            silhouette_avg_n_clusters.append(silhouette_avg)
            # Compute the silhouette scores for each sample
            sample_silhouette_values = silhouette_samples(X, cluster_labels)
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.set_size_inches(18, 7)

            # The 1st subplot is the silhouette plot
            # The silhouette coefficient can range from -1 to 1 but in this example lie within [-0.1, 1]
            ax1.set_xlim([-0.1, 1])
            # The (n_clusters +1)*10 is for inserting blank space between silhouette plots of individual clusters
            ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])



            y_lower = 10
            for i in range(n_clusters):
                # Aggregate the silhouette score for sample belonging to cluster i, and sort them
                ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.nipy_spectral(float(i) / n_clusters)
                ax1.fill_between(np.arange(y_lower, y_upper),
                                 0, ith_cluster_silhouette_values,
                                 facecolor=color, edgecolor=color, alpha=0.7)

                # Label the silhouette plots with their cluster numbers in the middle
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            ax1.set_title("The silhouette plot for the various clusters.")
            ax1.set_xlabel("The silhouette coefficient values")
            ax1.set_ylabel("Cluster label")

            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

            # 2nd plot showing the actual clusters formed
            colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
            ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                        c=colors, edgecolor='k')

            # Labeling the clusters
            centers = clusterer.cluster_centers_
            # Draw white circles at cluster centers
            ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                        c="white", alpha=1, s=200, edgecolor='k')

            for i, c in enumerate(centers):
                ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                            s=50, edgecolor='k')

            ax2.set_title("The visualization of the clustered data.")
            ax2.set_xlabel("Feature space for the 1st feature")
            ax2.set_ylabel("Feature space for the 2nd feature")

            plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                          "with n_clusters = %d" % n_clusters),
                         fontsize=14, fontweight='bold')

        plt.show()

        style.use("fivethirtyeight")
        plt.plot(range_clusters, silhouette_avg_n_clusters)
        plt.xlabel("Number of Clusters (k)")
        plt.ylabel("silhouette score")
        plt.show()

class HDBScan():

    def __init__(self, unlabeled_data):
        self.hdbscan = hdbscan.HDBSCAN()

            #HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True,
                                       #gen_min_span_tree=False, leaf_size=40, memory=Memory(cachedir=None),
                                       #metric='euclidean', min_cluster_size=5, min_samples=None, p=None)


        self.unlabeled_data_index = unlabeled_data[0]
        self.unlabeled_data = unlabeled_data[1]

    def hdbscan_fit(self):

        self.hdbscan.fit(self.unlabeled_data)

    def hdbscan_labels(self):

        return self.hdbscan.labels_

    def distance_sort(self, threshold: float = 1.0, n_instances: int = 100,
                     dist_measuring: str = 'euclidean'):

        x_val = self.unlabeled_data.copy()
        clusters = self.hdbscan_labels()

        x_val['original_index'] = self.unlabeled_data_index
        x_val['label_cluster'] = clusters
        # x_val[f'{percentile}th_percentile'] = np.percentile(distances,percentile)
        distribution_instances = round(n_instances / len(set(clusters)))

        distance_points = {}

        for i in list(set(clusters)):
            print('Cluster: ', i)
            temp_df = x_val[x_val['label_cluster'].isin([i])]
            # distance_points = np.empty((0, len(temp_df[0])), float)
            points = np.empty((0, temp_df.shape[1] - 3), float)

            for index, value in temp_df.iterrows():
                if points.shape[0] <= distribution_instances:
                    convert_series = value.to_frame().T
                    convert_series['original_index'] = convert_series['original_index'].astype(int)
                    convert_series['label_cluster'] = convert_series['label_cluster'].astype(int)
                    formulation_id = convert_series['original_index'].values[0]
                    data = convert_series.drop(columns=['distances', 'original_index', 'label_cluster'])
                    index = index

                    if points.shape[0] >= 1:
                        distance = cdist(points[-1:, :], data, dist_measuring)
                        if distance >= threshold:
                            distance_points[formulation_id] = distance[0][0]
                            points = np.append(points, data, axis=0)
                        # distance_points = np.append(distance_points, cdist(points[-1],data,'euclidean'))
                    else:
                        points = np.append(points, data, axis=0)

        print('Completed distance measuring...')

        distances_df = pd.DataFrame.from_dict(distance_points, orient='index')
        distances_df = distances_df.rename(columns={0: 'distances_local'})
        result = pd.merge(x_val, distances_df, left_on='original_index', right_index=True)
        results_index = result['original_index']
        distance_score = result['distances_local']
        result.drop(columns=['distances', 'original_index', 'label_cluster', 'distances_local'], inplace=True)

        return result, results_index, distance_score