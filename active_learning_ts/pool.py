from typing import Protocol

import tensorflow as tf


class Pool(Protocol):
    """this is the pool of possible query candidates"""
    def __init__(self):
        self.size = None
        self.shape = None

    def get_element(self, element: tf.Tensor) -> tf.Tensor:
        pass

    def get_shape(self):
        return self.shape

    def get_size(self):
        return self.size

    def get_elements(self, query_candidates):
        return [self.get_element(x) for x in query_candidates]
