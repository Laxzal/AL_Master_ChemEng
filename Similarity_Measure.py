import gower
import numpy as np
from scipy import spatial
import pandas as pd


class Similarity():

    def __init__(self, unlabeled_data, index_unlabeled_data):
        self.unlabeled_data = unlabeled_data
        self.index_unlabeled_data = index_unlabeled_data

    def similarity_cosine(self, threshold: float = 0.5, similarity_measurement: str = 'cosine', n_instances: int = 100):
        assert threshold <= 1.0
        unlabeled_data = pd.DataFrame(self.unlabeled_data.copy())
        unlabeled_data['original_index'] = self.index_unlabeled_data

        similarity_points = {}
        points = np.empty((0, unlabeled_data.shape[1] - 1), float)

        for index, value in unlabeled_data.iterrows():
            if points.shape[0] >= n_instances:
                convert_series = value.to_frame().T
                convert_series['original_index'] = convert_series['original_index'].astype(int)
                formulation_id = convert_series['original_index'].values[0]
                data = convert_series.drop(columns=['original_index'])
                index = index

                if points.shape[0] >= 1:
                    similarity = 1 - spatial.distance.cdist(points[-1:, :], data, metric=similarity_measurement)[0][0]
                    if similarity <= threshold:
                        similarity_points[formulation_id] = similarity
                        points = np.append(points, data, axis=0)
                else:
                    similarity_points[formulation_id] = 1.0
                    points = np.append(points, data, axis=0)
            else:
                break
        similarity_df = pd.DataFrame.from_dict(similarity_points, orient='index')
        similarity_df = similarity_df.rename(columns={0: 'similarity_score'})
        result = pd.merge(unlabeled_data, similarity_df, left_on='original_index', right_index=True)
        results_index = result['original_index']
        similarity_score = result['similarity_score']
        result.drop(columns=['original_index'], inplace=True)

        return result, results_index, similarity_score

    def similarity_gower(self, threshold, n_instances, converted_columns):

        assert threshold <= 1.0
        unlabeled_data = pd.DataFrame(self.unlabeled_data.copy())
        unlabeled_data['original_index'] = self.index_unlabeled_data

        similarity_points = {}
        points = np.empty((0, unlabeled_data.shape[1] - 1), float)

        for index, value in unlabeled_data.iterrows():
            if points.shape[0] <= n_instances:
                convert_series = value.to_frame().T
                convert_series['original_index'] = convert_series['original_index'].astype(int)
                formulation_id = convert_series['original_index'].values[0]
                data = convert_series.drop(columns=['original_index'])
                index = index

                if points.shape[0] >= 1:
                    points = np.append(points, data, axis=0)

                    # https://medium.com/analytics-vidhya/gowers-distance-899f9c4bd553
                    # The lower the value, the more similar it is. Therefore, 1 - gower makes it
                    # The lower the value the less similar it is.
                    similarity = 1 - gower.gower_matrix(points[[-2, -1]], cat_features=converted_columns)[1][0]

                    if similarity <= threshold:
                        print(str(index) + "/" + str(unlabeled_data.shape[0]) + " - Formulation ID: " + str(
                            formulation_id))
                        similarity_points[formulation_id] = similarity
                    else:
                        points = np.delete(points, -1, 0)
                else:
                    similarity_points[formulation_id] = 1.0
                    points = np.append(points, data, axis=0)
            else:
                break

        print("Completed Gower Scoring")
        print(
            "Found " + str(points.shape[0]) + " points with similarities less than or equal to: " + str(threshold))

        similarity_df = pd.DataFrame.from_dict(similarity_points, orient='index')
        similarity_df = similarity_df.rename(columns={0: 'similarity_score'})
        result = pd.merge(unlabeled_data, similarity_df, left_on='original_index', right_index=True)
        results_index = result['original_index']
        similarity_scores = result['similarity_score']
        result.drop(columns=['original_index', 'similarity_score'], inplace=True)
        result = result.to_numpy()
        return result, results_index, similarity_scores
