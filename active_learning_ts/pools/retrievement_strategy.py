from typing import List, Protocol

import tensorflow as tf

from active_learning_ts.data_retrievement.data_source import DataSource


class RetrievementStrategy(Protocol):
    def __init__(self):
        self.data_set = None
        self.data_source = None

    def post_init(self, data_source: DataSource):
        self.data_source = data_source

    def _find(self, point: tf.Tensor) -> List[tf.Tensor]:
        return [point]

    def find(self, points: List[tf.Tensor]) -> List[List[tf.Tensor]]:
        return [self._find(x) for x in points]
