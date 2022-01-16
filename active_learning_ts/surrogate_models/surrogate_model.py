from typing import Protocol, Tuple

import tensorflow as tf

from active_learning_ts.data_retrievement.data_retriever import DataRetriever
from active_learning_ts.pool import Pool
from active_learning_ts.pools.continuous_vector_pool import ContinuousVectorPool


class SurrogateModel(Protocol):
    """
    The goal of a SurrogateModel is to as best as possible, emulate the Data Retrievement process. What constitutes a
    good emulation of the Data Retrievement process may be model/use-case specific
    """

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
        pass

    def learn(self, points: tf.Tensor, feedback: tf.Tensor):
        """
        Trains the model at the given points using the given feedback

        :param points: the points queried
        :param feedback: the feedback to be used for training
        :return: model specific
        """
        pass

    def query(self, points: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Atempts to simulate the data source at the given points
        :param points: 2D
        :return: 2D
        """
        pass

    def get_query_pool(self) -> Pool:
        if not self.query_pool.is_discrete():
            return self.query_pool
        else:
            self.query_pool = ContinuousVectorPool(self.query_pool.shape[0], ranges=self.query_pool.get_ranges())
            return self.query_pool
