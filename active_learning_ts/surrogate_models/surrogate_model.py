from typing import Protocol, Tuple

import tensorflow as tf

from active_learning_ts.pool import Pool


class SurrogateModel(Protocol):
    """
    The goal of a SurrogateModel is to as best as possible, emulate the Data Retrievement process. What constitutes a
    good emulation of the Data Retrievement process may be model/use-case specific
    """

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
        pass
