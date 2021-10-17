from typing import Protocol, List
import tensorflow as tf


class SelectionCriteria(Protocol):

    def score_queries(self, queries: List[tf.Tensor]) -> tf.Tensor:
        pass
