import numpy as np
import torch
import torch.nn.functional as F
from random import shuffle


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
            print (added)
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

        item_tensor = torch.FloatTensor(data)
        center_tensor = torch.FloatTensor(center_vec)

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
