from typing import Protocol, List, Tuple

import tensorflow as tf


class Pool(Protocol):
    """this is the pool of possible query candidates"""

    shape: Tuple[int]
    ranges: List[List[Tuple[float, float]]]

    def __init__(self):
        ...

    def get_element_normalized(self, element: tf.Tensor) -> tf.Tensor:
        """
        Gets the point using the given retrievement strategy. The pool is considered to be normalized onto the range
        [0,1]. How this normalisation is done is pool specific.

        :param: element, a tensor with values in the range [0,1]
        :return: the corresponding index
        """
        ...

    def get_shape(self):
        return self.shape

    def get_elements_normalized(self, query_candidates) -> tf.Tensor:
        """
        Gets the point using the given retrievement strategy. The pool is considered to be normalized onto the range
        [0,1]. How this normalisation is done is pool specific

        :param: element, a list of tensors with values in the range [0,1]
        :return: the corresponding index
        """
        return tf.convert_to_tensor([self.get_element_normalized(x) for x in query_candidates])

    def get_elements(self, query_candidates: tf.Tensor) -> tf.Tensor:
        """
        Gets the point using the given retrievement strategy.

        :param: element, a list of tensors that we want to query
        :return: the corresponding index points that we can query
        """
        ...

    def normalize(self, query_candidates: tf.Tensor) -> tf.Tensor:
        """
        Normalizes the given list of vectors onto [0,1]^n
        """
        return tf.map_fn(lambda x: self._normalize(x), query_candidates)

    def _normalize(self, query_candidate: tf.Tensor) -> tf.Tensor:
        ...

    def is_discrete(self) -> bool:
        """
        Returns whether this pool is discrete. If it is discrete, then it defines the method get_all_elements
        """
        return False

    def get_all_elements(self) -> List[tf.Tensor]:
        """
        Returns a list of all elements in the given pool. This is only possible if is_discrete evaluates to true
        """
        ...

    def get_elements_with_index(self, indices: tf.Tensor):
        """
        Gets points corresponding to the given indices
        """
        return indices

    def get_ranges(self):
        """
        Overwrite if the ranges are not stored in a variable called 'ranges'

        :return:
        """
        return self.ranges

    def is_valid(self, point) -> bool:
        """
        Checks if the given point is in the pool
        """
        return True
