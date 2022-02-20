from sklearn.cluster import KMeans
"""
1. Cluster the data
2. Estimate P(y|k)
3. Calculate P(y|x)
4. Choose unlabeled sample based on Eq1 and label
5. Re-cluster if necessary
6. Repeat steps until stop
"""


#1. Cluster the data


kmeans = KMeans(n_clusers=5)