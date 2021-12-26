from dataclasses import dataclass

import tensorflow as tf
from typing_extensions import Protocol


@dataclass()
class DataInstance:
    query_candidates = tf.convert_to_tensor([])
    actual_queries = tf.convert_to_tensor([])
    query_results = tf.convert_to_tensor([])
    quality = tf.convert_to_tensor([])
    cost = tf.convert_to_tensor([])


class DataInstanceFactory(Protocol):
    def __call__(self) -> DataInstance:
        return DataInstance()
