from typing import Protocol
import tensorflow as tf


class QuerySampler(Protocol):
    def __init__(self):
        self.pool = None

    def post_init(self, pool):
        self.pool = pool

    def sample(self, num_queries: int = 1) -> tf.Tensor:
        pass
