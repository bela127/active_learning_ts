
from active_learning_ts.data_retrievement.data_source import DataSource
from active_learning_ts.pools.continuous_vector_pool import ContinuousVectorPool
from active_learning_ts.pools.retrievement_strategy import RetrievementStrategy

import tensorflow as tf
from scipy.spatial import KDTree


class NearestNeighboursFindStrategy(RetrievementStrategy):
    """
    Allows discrete Pool to be referenced as a continuous pool
    """

    def __init__(self, num_neighbours: int):
        self.num_neighbours = num_neighbours
        self.kd_tree = None
        self.data_set = None
        self.continuous_pool: ContinuousVectorPool

    def post_init(self, data_source: DataSource):
        if not data_source.possible_queries().is_discrete():
            raise TypeError("Nearest Neighbour requires a discrete pool")
        self.data_source = data_source
        self.data_set = data_source.possible_queries().get_all_elements()
        self.kd_tree = KDTree(self.data_set)

        ranges = []

        for i in range(self.data_source.possible_queries().shape[0]):
            minimum = self.data_set[0][i]
            maximum = self.data_set[0][i]
            for j in range(1, len(self.data_set)):
                minimum = min(minimum, self.data_set[j][i])
                maximum = max(maximum, self.data_set[j][i])

            ranges.append([(minimum, maximum)])
        self.continuous_pool = ContinuousVectorPool(dim=self.data_source.possible_queries().shape[0], ranges=ranges)

    def find(self, points: tf.Tensor) -> tf.Tensor:
        """

        :param point: 1D Tensor (vector)
        :return: 1D Indices to query
        """
        return tf.convert_to_tensor([self.kd_tree.query(point, k=self.num_neighbours)[1] for point in points], dtype=tf.dtypes.int32)

    def get_query_pool(self):
        return self.continuous_pool
