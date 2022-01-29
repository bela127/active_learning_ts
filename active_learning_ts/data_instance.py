from dataclasses import dataclass

import tensorflow as tf
from typing_extensions import Protocol


@dataclass()
class DataInstance:
    query_candidates: tf.Tensor = tf.convert_to_tensor([])
    actual_queries: tf.Tensor = tf.convert_to_tensor([])
    query_results: tf.Tensor = tf.convert_to_tensor([])
    quality: tf.Tensor = tf.convert_to_tensor([])
    cost: tf.Tensor = tf.convert_to_tensor([])


class DataInstanceFactory(Protocol):
    def __call__(self) -> DataInstance:
        return DataInstance()
