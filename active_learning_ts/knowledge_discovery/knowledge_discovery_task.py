from typing import Protocol

import tensorflow as tf

from active_learning_ts.query_selection.query_sampler import QuerySampler
from active_learning_ts.pipeline_element import PipelineElement


class KnowledgeDiscoveryTask(Protocol):
    """
    The goal of a KnowledgeDiscoveryTask is to as best as possible, is to learn something about the given data.
    It uses the Surrogate Model to emulate the data, and learns from the data provided by the SurrogateModel.
    """
    surrogate_model: PipelineElement
    sampler: QuerySampler

    def post_init(self, surrogate_model: PipelineElement, sampler: QuerySampler):
        self.surrogate_model = surrogate_model
        self.sampler = sampler

    def uncertainty(self, points: tf.Tensor) -> tf.Tensor:
        """
        Returns the uncertainty of the model at the given points. A higher number means the model is less certain
        :param points: the points at which the uncertainty should be measured
        :return: the uncertainties as a tensor.
        """
        ...

    def learn(self, num_queries):
        """
        When this method is called, the knowledge discovery task will use the Surrogate Model to generate data and learn
        from it.
        Whether it resets every time, or retains information learned from previous steps is implementation specific

         :return: model specific
        """
        ...
