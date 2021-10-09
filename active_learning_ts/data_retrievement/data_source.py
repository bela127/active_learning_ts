from typing import Protocol, List, Tuple
import tensorflow as tf


class DataSource(Protocol):
    def __init__(self) -> None:
        pass

    def query(self, actual_queries: List[tf.Tensor]) -> List[Tuple[tf.Tensor, tf.Tensor]]:
        pass

    def possible_queries(self):
        pass
