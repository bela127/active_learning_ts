from typing import Protocol, Tuple

import tensorflow as tf

from active_learning_ts.data_retrievement.data_retriever import DataRetriever
from active_learning_ts.data_retrievement.pool import Pool
from active_learning_ts.queryable import Queryable


class SurrogateModel(Queryable, Protocol):
    """
    The goal of a SurrogateModel is to as best as possible, emulate the Data Retrievement process. What constitutes a
    good emulation of the Data Retrievement process may be model/use-case specific
    """
    query_pool: Pool
    point_shape: Tuple
    value_shape: Tuple

    def post_init(self, data_retriever: DataRetriever):
        self.query_pool = data_retriever.get_query_pool()
        self.point_shape = data_retriever.point_shape
        self.value_shape = data_retriever.value_shape

    def uncertainty(self, points: tf.Tensor) -> tf.Tensor:
        """
        Returns the uncertainty of the model at the given points. A higher number means the model is less certain
        :param points: the points at which the uncertainty should be measured
        :return: the uncertainties as a tensor.
        """
        ...

    def learn(self, points: tf.Tensor, feedback: tf.Tensor):
        """
        Trains the model at the given points using the given feedback

        :param points: the points queried
        :param feedback: the feedback to be used for training
        :return: model specific
        """
        ...

    def query(self, points: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Atempts to simulate the data source at the given points
        :param points: 2D
        :return: 2D
        """
        ...

    def get_query_pool(self) -> Pool:
        return self.query_pool
