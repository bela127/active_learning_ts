from typing import Protocol

import tensorflow as tf

from active_learning_ts.data_retrievement.pool import Pool


class RetrievementStrategy(Protocol):
    pool: Pool

    def __init__(self):
        ...

    def post_init(self, pool: Pool):
        self.pool = pool

    def find(self, points: tf.Tensor) -> tf.Tensor:
        """
        Finds valid points in the pool given the point that we are trying to query. For example, this could find the
        closest point in the dataset that we can query.

        :param: points,  a list of points that we want to query
        :return: a list of lists of points. Each list of points is the result of the querying of a single point.
        """
        if points.shape.rank == 2:
            return points
        else:
            return tf.reshape(points, (1, points.shape[0]))

    def get_query_pool(self):
        return self.pool
