import tensorflow as tf
from scipy.spatial import KDTree

from active_learning_ts.pool import Pool
from active_learning_ts.pools.continuous_vector_pool import ContinuousVectorPool
from active_learning_ts.pools.retrievement_strategy import RetrievementStrategy


class NearestNeighboursFindStrategy(RetrievementStrategy):
    """
    Allows discrete Pool to be referenced as a continuous pool
    """

    def __init__(self, num_neighbours: int):
        self.num_neighbours = num_neighbours
        self.kd_tree = None
        self.data_set = None
        self.continuous_pool: ContinuousVectorPool

    def post_init(self, pool: Pool):
        if not pool.is_discrete():
            raise TypeError("Nearest Neighbour requires a discrete pool")

        self.pool = pool
        self.data_set = pool.get_all_elements()
        self.kd_tree = KDTree(self.data_set)

        ranges = []

        for i in range(self.pool.shape[0]):
            minimum = self.data_set[0][i]
            maximum = self.data_set[0][i]
            for j in range(1, len(self.data_set)):
                minimum = min(minimum, self.data_set[j][i])
                maximum = max(maximum, self.data_set[j][i])

            ranges.append([(minimum, maximum)])
        self.continuous_pool = ContinuousVectorPool(dim=self.pool.shape[0], ranges=ranges)

    def find(self, points: tf.Tensor) -> tf.Tensor:
        """

        :param points: 2D Tensor (a collection of vectors)
        :return: 2D Indices to query (batches of queries)
        """
        return tf.convert_to_tensor([self.kd_tree.query(point, k=self.num_neighbours)[1] for point in points],
                                    dtype=tf.dtypes.int32)

    def get_query_pool(self):
        return self.continuous_pool
