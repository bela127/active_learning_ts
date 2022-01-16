from typing import Protocol, Tuple

import tensorflow as tf

from active_learning_ts.pool import Pool


class DataSource(Protocol):
    """
    Generates data
    """

    def __init__(self) -> None:
        self.retrievementStrategy = None
        self.point_shape = None
        self.value_shape = None

    def query(self, actual_queries: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """

        :param actual_queries: 2D Tensor
        :return: 2 2D Tensors
        """
        pass

    def possible_queries(self) -> Pool:
        pass

    def post_init(self, retrievement_strategy):
        self.retrievementStrategy = retrievement_strategy
