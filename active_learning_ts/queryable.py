from typing import Protocol, Tuple

import tensorflow as tf

from active_learning_ts.data_retrievement.pool import Pool


class Queryable(Protocol):
    """
    A class representing an object that can be queried. It can be given queris, for which it can return results
    """

    """
    The required shape of the query
    """
    point_shape: Tuple
    """
    The shape of the result
    """
    value_shape: Tuple

    def __init__(self):
        ...

    def query(self, query : tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        A query function. The input is a tensor (list) of queries, each of which has shape point_shape.
        The ouput is a Tuple (x,y), where x is a list of the queries that were answered (not necessarily a sublist of
        query), and y is a list of the results of those queries (respective to the order of x)
        """
        ...

    def get_query_pool(self) -> Pool:
        """
        Returns a pool of possible queries
        """
        ...
