from typing import List, Protocol, Tuple, Any

import tensorflow as tf

from active_learning_ts.pool import Pool


class RetrievementStrategy(Protocol):
    def __init__(self):
        self.pool: Pool = None

    def post_init(self, pool: Pool):
        self.pool = pool

    def find(self, points: tf.Tensor) -> tf.Tensor:
        """

        :param points: a 2D tensor
        :return: a 3D tensor
        """
        if points.shape.rank == 2:
            return points
        else:
            return tf.reshape(points, (1, points.shape[0]))

    def get_query_pool(self):
        return self.pool
