from typing import Protocol, Tuple

import tensorflow as tf

from active_learning_ts.pool import Pool


# TODO make datasource implement this
class Queryable(Protocol):
    point_shape: Tuple
    value_shape: Tuple

    def __init__(self):
        ...


    def query(self, query) -> Tuple[tf.Tensor, tf.Tensor]:
        ...

    def get_query_pool(self) -> Pool:
        ...
