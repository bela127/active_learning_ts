from typing import Protocol

import tensorflow as tf


class QuerySampler(Protocol):
    def __init__(self):
        self.pool = None

    def post_init(self, pool):
        """
        You need to be able to call this multiple times
        :param pool:
        :return:
        """
        self.pool = pool

    def sample(self, num_queries: int = 1) -> tf.Tensor:
        pass

    def update_pool(self, pool):
        raise NotImplementedError("Cannot dynamically change pool")
