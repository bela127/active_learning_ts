from typing import Protocol, List, Tuple

import tensorflow as tf


class Pool(Protocol):
    """this is the pool of possible query candidates"""

    shape: Tuple[int]
    ranges: Tuple

    def __init__(self):
        ...

    def get_element_normalized(self, element: tf.Tensor) -> tf.Tensor:
        ...

    def get_shape(self):
        return self.shape

    def get_elements_normalized(self, query_candidates) -> tf.Tensor:
        return tf.convert_to_tensor([self.get_element_normalized(x) for x in query_candidates])

    def get_elements(self, query_candidates: tf.Tensor) -> tf.Tensor:
        ...

    def normalize(self, query_candidates: tf.Tensor) -> tf.Tensor:
        return tf.map_fn(lambda x: self._normalize(x), query_candidates)

    def _normalize(self, query_candidate: tf.Tensor) -> tf.Tensor:
        ...

    def is_discrete(self) -> bool:
        return False

    def get_all_elements(self) -> List[tf.Tensor]:
        ...

    def get_elements_with_index(self, indices: tf.Tensor):
        return indices

    def get_ranges(self):
        """
        Overwrite if the ranges are not stored in a variable called 'ranges'

        :return:
        """
        return self.ranges

    def is_valid(self, point) -> bool:
        return True
