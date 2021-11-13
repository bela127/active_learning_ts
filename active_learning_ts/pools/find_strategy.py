from typing import List, Protocol

import tensorflow as tf


class FindStrategy(Protocol):
    def __init__(self):
        self.data_set = None

    def _find(self, point: tf.Tensor) -> List[tf.Tensor]:
        pass

    def find(self, points: List[tf.Tensor]) -> List[List[tf.Tensor]]:
        return [self._find(x) for x in points]

    def post_init(self, data_set):
        pass
