from typing import Protocol, List, Tuple
import tensorflow as tf


class DataSource(Protocol):
    """
    Generates data
    """
    def __init__(self) -> None:
        pass

    def query(self, actual_queries: List[tf.Tensor]) -> Tuple[List[tf.Tensor], List[tf.Tensor]]:
        pass

    def possible_queries(self):
        pass

    def get_query_pool(self):
        pass
