from typing import Protocol, Tuple

import tensorflow as tf

from active_learning_ts.data_retrievement.pool import Pool
from active_learning_ts.data_retrievement.retrievement_strategy import RetrievementStrategy


class DataSource(Protocol):
    """
    Generates data
    """
    retrievement_strategy: RetrievementStrategy
    point_shape: Tuple
    value_shape: Tuple

    def __init__(self) -> None:
        ...

    def query(self, actual_queries: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """

        :param actual_queries: 2D Tensor
        :return: 2 2D Tensors
        """
        ...

    def possible_queries(self) -> Pool:
        ...

    def post_init(self, retrievement_strategy: RetrievementStrategy):
        self.retrievement_strategy = retrievement_strategy
