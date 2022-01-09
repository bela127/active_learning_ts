from typing import Protocol, List

import tensorflow as tf


class Pool(Protocol):
    """this is the pool of possible query candidates"""

    def __init__(self):
        self.shape = None

    def _get_element_normalized(self, element: tf.Tensor) -> List[tf.Tensor]:
        pass

    def get_shape(self):
        return self.shape

    def get_elements_normalized(self, query_candidates) -> tf.Tensor:
        return tf.convert_to_tensor([self._get_element_normalized(x) for x in query_candidates])

    def get_elements(self, query_candidates: tf.Tensor) -> tf.Tensor:
        pass

    def normalize(self, query_candidates: tf.Tensor) -> tf.Tensor:
        return tf.map_fn(lambda x: self._normalize(x), query_candidates)

    def _normalize(self, query_candidate: tf.Tensor) -> tf.Tensor:
        pass

    def is_discrete(self) -> bool:
        return False

    def get_all_elements(self) -> List[tf.Tensor]:
        pass

    def get_elements_with_index(self, indices: tf.Tensor):
        return indices
