from __future__ import annotations
import imp
from typing import TYPE_CHECKING

from active_learning_ts.data_retrievement.data_source import DataSource
from active_learning_ts.data_retrievement.pools.continuous_vector_pool import ContinuousVectorPool

if TYPE_CHECKING:
    import tensorflow as tf
    from typing import Tuple
    from active_learning_ts.data_retrievement.pool import Pool

class TestDataSource(DataSource):
    def __init__(self):
        self.value_shape = (1,)
        self.point_shape = (1,)
    
    def query(self, actual_queries: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """

        :param actual_queries: 2D Tensor
        :return: 2 2D Tensors
        """
        return actual_queries, actual_queries

    def possible_queries(self) -> Pool:
        return None
