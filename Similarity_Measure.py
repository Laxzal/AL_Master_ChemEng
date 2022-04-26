import numpy as np
from scipy import spatial
import pandas as pd


class Similarity():

    def __init__(self, unlabeled_data, index_unlabeled_data):
        self.unlabeled_data = unlabeled_data
        self.index_unlabeled_data = index_unlabeled_data

    def similarity(self, threshold: float = 0.5, similarity_measurement: str = 'cosine'):
        assert threshold <= 1.0
        unlabeled_data = pd.DataFrame(self.unlabeled_data.copy())
        unlabeled_data['original_index'] = self.index_unlabeled_data

        similarity_points = {}
        points = np.empty((0, unlabeled_data.shape[1] - 1), float)

        for index, value in unlabeled_data.iterrows():
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

        similarity_df = pd.DataFrame.from_dict(similarity_points, orient='index')
        similarity_df = similarity_df.rename(columns={0: 'similarity_score'})
        result = pd.merge(unlabeled_data, similarity_df, left_on='original_index', right_index=True)
        results_index = result['original_index']
        result.drop(columns=['original_index'], inplace=True)

        return result, results_index
