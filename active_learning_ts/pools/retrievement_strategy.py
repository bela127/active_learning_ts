from typing import List, Protocol

import tensorflow as tf

from active_learning_ts.data_retrievement.data_source import DataSource


class RetrievementStrategy(Protocol):
    # TODO: for discrete find strategies. it would be a lot more efficient, if they returned the index
    def __init__(self):
        self.data_source: DataSource = None

    def post_init(self, data_source: DataSource):
        self.data_source = data_source

    def _find(self, point: tf.Tensor) -> tf.Tensor:
        return point

    def find(self, points: tf.Tensor) -> tf.Tensor:
        """

        :param points: a 2D tensor
        :return: a 3D tensor
        """
        # TODO: use map here instead
        return tf.convert_to_tensor([self._find(x) for x in points])

    def get_query_pool(self):
        return self.data_source.possible_queries()
