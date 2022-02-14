from typing import List
import tensorflow as tf
from scipy.spatial import KDTree

from active_learning_ts.data_retrievement.pool import Pool
from active_learning_ts.data_retrievement.pools.continuous_vector_pool import ContinuousVectorPool
from active_learning_ts.data_retrievement.retrievement_strategy import RetrievementStrategy


class NearestNeighboursRetrievementStrategy(RetrievementStrategy):
    """
    Allows discrete Pool to be referenced as a continuous pool
    """

    def __init__(self, num_neighbours: int):
        self.num_neighbours = num_neighbours
        self.kd_tree: KDTree
        self.all_elements: List[tf.Tensor]
        self.continuous_pool: ContinuousVectorPool

    def post_init(self, pool: Pool):
        if not pool.is_discrete():
            raise TypeError("Nearest Neighbour requires a discrete pool")

        self.pool = pool
        self.all_elements = pool.get_all_elements()
        self.kd_tree = KDTree(self.all_elements)

        ranges = []

        for i in range(self.pool.shape[0]):
            minimum = self.all_elements[0][i]
            maximum = self.all_elements[0][i]
            for j in range(1, len(self.all_elements)):
                minimum = min(minimum, self.all_elements[j][i])
                maximum = max(maximum, self.all_elements[j][i])

            ranges.append(((minimum, maximum),))

        self.continuous_pool = ContinuousVectorPool(
            dim=self.pool.shape[0], ranges=tuple(ranges)
        )

    def find(self, points: tf.Tensor) -> tf.Tensor:
        """

        :param points: 2D Tensor (a collection of vectors)
        :return: 2D Indices to query (batches of queries)
        """
        return tf.convert_to_tensor(
            [self.kd_tree.query(point, k=self.num_neighbours)[1] for point in points],
            dtype=tf.dtypes.int32,
        )

    def get_query_pool(self):
        return self.continuous_pool
