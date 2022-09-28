import gower
import numpy as np
from scipy import spatial
import pandas as pd
from datetime import datetime

class Similarity():

    def __init__(self, unlabeled_data, index_unlabeled_data):
        self.unlabeled_data = unlabeled_data
        self.index_unlabeled_data = index_unlabeled_data
        self.points = None

    def _loop_function(self, threshold, unlabeled_data, similarity_points, n_instances, converted_columns,
                       threshold_level):

        '''

        The _loop_function determines the gower measurement of each data point compared to the points array created in
        the function similarity_gower.

        The core of the function is as follows:

        The max ranked standard deviation unlabelled dataset is iterated over using the method .iterrows(). This
        method returns the index and the row (as a series) as pairs. If the points array is empty, then the first max
        ranked std deviation unlabelled formulation is appended to the points array. Once the points array has its
        first data point appended, the next max ranked std unlabelled formulation is compared to the first data point
        in the points array via the gower method. If the gower measurement returns a value less than the threshold,
        it means the data point passes the threshold requirement of dissimilarity and is added to the points array.
        However, if it is higher than the threshold, it is subsequently removed.

        Now, if there are, for example, two data points in the points array, the next max std deviation unlabelled
        formulation is compared to both of the points within the points array. If it's dissimilarity scores for each
        data points in the point are lower than the threshold at the time, then it is added to the points array.
        However, if it's dissimilarity score to one of the data points in the points array is higher than the
        threshold, it is subsequently removed.

        Overall, this row iteration continues until all formulations in the max ranked std deviation unlabelled dataset
        has been compared.

        The function then returns the selected unlabelled formulations that passed the threshold, their index in the
        unlabelled dataset, their similarity scores and the points array (which may or may not be higher than the
        required number of chosen instances set by the user).


        :param threshold: float, a value set by the user that determines whether the unlabelled formulation is
        dissimilar enough to be considered for labelling
        :param unlabeled_data: dataframe, the unlabelled dataset that has been ranked via the max standard deviation
        sampling strategy
        :param similarity_points: dict, a dictionary of the formulation index and their similarity score
        :param n_instances: int, number of required unlabelled formulations to be selected, set by the user
        :param converted_columns: list, the feature names
        :param threshold_level: dict, a dictionary of the threshold level the unlabelled formulation was chosen at the
        time
        :return: The function then returns the selected unlabelled formulations that passed the threshold, their index
        in the unlabelled dataset, their similarity scores and the points array (which may or may not be higher than the
        required number of chosen instances set by the user).
        '''
        now = datetime.now()

        current_time = now.strftime("%H:%M:%S")
        print(f" {current_time} - Starting Gower Scoring for {threshold}")
        for index, value in unlabeled_data.iterrows():
            if self.points.shape[0] < n_instances:
                convert_series = value.to_frame().T
                convert_series['original_index'] = convert_series['original_index'].astype(int)
                formulation_id = convert_series['original_index'].values[0]
                data = convert_series.drop(columns=['original_index'])
                index = index

                if self.points.shape[0] >= 1:


                    # https://medium.com/analytics-vidhya/gowers-distance-899f9c4bd553
                    # The lower the value, the more similar it is. Therefore, 1 - gower makes it
                    # The lower the value the less similar it is.
                    similarity = 1 - gower.gower_matrix(self.points,data, cat_features=converted_columns).min(axis=0)

                    if similarity <= threshold:
                        self.points = np.append(self.points, data, axis=0)
                        print(str(index) + "/" + str(unlabeled_data.shape[0]) + " - Formulation ID: " + str(
                            formulation_id))
                        similarity_points[formulation_id] = similarity
                        threshold_level[formulation_id] = threshold
                else:
                    similarity_points[formulation_id] = 0
                    self.points = np.append(self.points, data, axis=0)
                    threshold_level[formulation_id] = threshold
                    print(str(index) + "/" + str(unlabeled_data.shape[0]) + " - Formulation ID: " + str(
                        formulation_id))
            else:
                break
        now = datetime.now()

        current_time = now.strftime("%H:%M:%S")
        print(f" {current_time} - Completed Gower Scoring")
        print(
            "Found " + str(self.points.shape[0]) + " points with similarities less than or equal to: " + str(threshold))

        similarity_df = pd.DataFrame.from_dict(similarity_points, orient='index')
        similarity_df = similarity_df.rename(columns={0: 'similarity_score'})
        result = pd.merge(unlabeled_data, similarity_df, left_on='original_index', right_index=True)
        results_index = result['original_index']
        similarity_scores = result['similarity_score']
        result.drop(columns=['original_index', 'similarity_score'], inplace=True)
        result = result.to_numpy()

        return result, results_index, similarity_scores, self.points

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
        '''
        The Gower Measurement determines how similar each data point is to each other. In this function, the ranked
        standard deviation of the unlabelled dataset are compared with each other.

        The function begins with copying the ranked max standard deviation unlabelled dataset. Then, a dictionary
        variable, similarity_points, is initialised. After this, the empty 2-D array called points,
        with the dimension (0, [number of rows in the unlabelled dataset] -1) is also initialized. Finally,
        the variable threshold_level is also initialised.

        A while loop then begins, with the condition that if the number of data points in the variable points is less
        than the number of instances chosen by the user, the function in the while loop will continue.

        The _loop_function returns the results of the gower measurement, the index of the results, their similarity
        scores and the points array. If the points array is less than the number of required selected instances set
        by the user then the threshold level is increase by 0.1. The _loop_function then occurs again with a higher
        threshold than previously used.

        Eventually, once the points variable array has enough data points, the while loop ends and the function
        similarity_gower returns the selected max rank unlabelled samples for the user to label.



        :param threshold: float, the initial threshold level set by the user before incrementation
        :param n_instances: int, the required number of selected unlabelled formulations required by the user
        :param converted_columns: list, a list of the feature names
        :return: the function similarity_gower returns the selected max rank unlabelled samples for the user to label
        determined by the gower measurement and n_instances
        '''

        assert threshold <= 1.0
        unlabeled_data = pd.DataFrame(self.unlabeled_data.copy())
        unlabeled_data['original_index'] = self.index_unlabeled_data

        similarity_points = {}
        self.points = np.empty((0, unlabeled_data.shape[1] - 1), float)
        threshold_level = {}

        while self.points.shape[0] < n_instances:
            result, results_index, similarity_scores, self.points = self._loop_function(threshold, unlabeled_data,
                                                                                        similarity_points,
                                                                           n_instances,converted_columns,
                                                                           threshold_level)

            if threshold <= 1.0:
                threshold += 0.1

        return result, results_index, similarity_scores
