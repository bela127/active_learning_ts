from typing import Protocol, List
import tensorflow as tf


class SelectionCriteria(Protocol):
    def generate_queries(self, num_queries: int = 1) -> List[tf.Tensor]:
        pass

    def inform(self, queries: List[tf.Tensor], query_results: List[tf.Tensor]) -> None:
        pass
