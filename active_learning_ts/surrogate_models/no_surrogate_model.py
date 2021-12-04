from typing import List

from active_learning_ts.data_retrievement.data_retriever import DataRetriever
from active_learning_ts.surrogate_models.surrogate_model import SurrogateModel
import tensorflow as tf


class NoSurrogateModel(SurrogateModel):
    def __init__(self):
        self.data_retriever: DataRetriever = None

    def uncertainty(self, points: tf.Tensor) -> tf.Tensor:
        return tf.convert_to_tensor([.0] * len(points))

    def query(self, points: tf.Tensor) -> tf.Tensor:
        return self.data_retriever.retrieve(points)[1]

    def post_init(self, data_retriever):
        self.data_retriever = data_retriever
