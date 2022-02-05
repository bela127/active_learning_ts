from typing import Tuple

import tensorflow as tf

from active_learning_ts.data_retrievement.data_retriever import DataRetriever
from active_learning_ts.pool import Pool
from active_learning_ts.surrogate_models.surrogate_model import SurrogateModel


class NoSurrogateModel(SurrogateModel):
    def __init__(self):
        self.data_retriever: DataRetriever = None

    def uncertainty(self, points: tf.Tensor) -> tf.Tensor:
        return tf.convert_to_tensor([.0] * len(points))

    def query(self, points: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        return self.data_retriever.query(points)

    def post_init(self, data_retriever):
        super(NoSurrogateModel, self).post_init(data_retriever)
        self.data_retriever = data_retriever

    def get_query_pool(self) -> Pool:
        return self.data_retriever.get_query_pool()
