from typing import Protocol

import tensorflow as tf

from active_learning_ts.data_retrievement.pool import Pool


class QuerySampler(Protocol):
    """
    A query sampler is used to generate points that can be used to sample data.
    """

    def __init__(self):
        self.pool: Pool = None

    def post_init(self, pool):
        """
        You need to be able to call this multiple times
        :param pool:
        :return:
        """
        self.pool = pool

    def sample(self, num_queries: int = 1) -> tf.Tensor:
        """
        Returns a point that can be queried, according to the given pool. The returned value is referred to as an index-
        point. The shape and information of what an index-point looks like is decided by the retrievement strategy of
        the given pool
        """
        pass

    def update_pool(self, pool):
        raise NotImplementedError("Cannot dynamically change pool")
