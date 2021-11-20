from typing import Protocol, List

import tensorflow as tf


class Pool(Protocol):
    """this is the pool of possible query candidates"""

    def __init__(self):
        self.size = None
        self.shape = None

    def _get_element_normalized(self, element: tf.Tensor) -> List[tf.Tensor]:
        pass

    def get_shape(self):
        return self.shape

    def get_size(self):
        return self.size

    def get_elements_normalized(self, query_candidates) -> List[List[tf.Tensor]]:
        return [self._get_element_normalized(x) for x in query_candidates]

    def _get_element(self, element: tf.Tensor) -> List[tf.Tensor]:
        # TODO: implement this in subclasses
        pass

    def get_elements(self, query_candidates: List[tf.Tensor]) -> List[List[tf.Tensor]]:
        return [self._get_element(x) for x in query_candidates]

    def normalize(self, query_candidates: List[tf.Tensor]) -> List[tf.Tensor]:
        return [self._normalize(x) for x in query_candidates]

    def _normalize(self, query_candidate: tf.Tensor) -> tf.Tensor:
        pass

    def is_discrete(self) -> bool:
        return False
