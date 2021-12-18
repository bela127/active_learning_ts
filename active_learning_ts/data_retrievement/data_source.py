from typing import Protocol, List, Tuple
import tensorflow as tf

from active_learning_ts.pool import Pool


class DataSource(Protocol):
    """
    Generates data
    """

    def __init__(self) -> None:
        self.retrievementStrategy = None

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
