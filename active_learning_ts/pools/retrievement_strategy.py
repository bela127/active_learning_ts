from typing import List, Protocol

import tensorflow as tf

from active_learning_ts.data_retrievement.data_source import DataSource


class RetrievementStrategy(Protocol):
    # TODO: for discrete find strategies. it would be a lot more efficient, if they returned the index
    def __init__(self):
        self.data_source: DataSource = None

    def post_init(self, data_source: DataSource):
        self.data_source = data_source

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
        return self.data_source.possible_queries()
