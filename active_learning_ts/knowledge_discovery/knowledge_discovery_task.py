from typing import Protocol, List
import tensorflow as tf

from active_learning_ts.pool import Pool
from active_learning_ts.surrogate_models.surrogate_model import SurrogateModel


class KnowledgeDiscoveryTask(Protocol):
    """
    The goal of a KnowledgeDiscoveryTask is to as best as possible, is to learn something about the given data.
    It uses the Surrogate Model to emulate the data, and learns from the data provided by the SurrogateModel.
    """

    def __init__(self):
        self.surrogate_model: SurrogateModel = None
        self.surrogate_pool: Pool = None

    def post_init(self, surrogate_model: SurrogateModel, surrogate_pool: Pool):
        self.surrogate_model = surrogate_model
        self.surrogate_pool = surrogate_pool

    def uncertainty(self, points: List[tf.Tensor]) -> tf.Tensor:
        """
        Returns the uncertainty of the model at the given points. A higher number means the model is less certain
        :param points: the points at which the uncertainty should be measured
        :return: the uncertainties as a tensor.
        """
        pass

    def learn(self):
        """
        When this method is called, the knowledge discovery task will use the Surrogate Model to generate data and learn
        from it.
        Whether it resets every time, or retains information learned from previous steps is implementation specific

         :return: model specific
        """
        pass
