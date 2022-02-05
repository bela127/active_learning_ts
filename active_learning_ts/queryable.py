from typing import Protocol, Tuple

import tensorflow as tf

from active_learning_ts.pool import Pool


# TODO make datasource implement this
class Queryable(Protocol):
    def __init__(self):
        self.point_shape = None
        self.value_shape = None

    def query(self, query) -> Tuple[tf.Tensor, tf.Tensor]:
        pass

    def get_query_pool(self) -> Pool:
        pass
