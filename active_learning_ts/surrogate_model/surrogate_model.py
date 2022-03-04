from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass
from typing import Protocol

import tensorflow as tf

from active_learning_ts.experiments.blueprint_config import BlueprintConfig
from active_learning_ts.pipeline_element import PipelineElement

if TYPE_CHECKING:
    from typing import Tuple, Type
    from active_learning_ts.data_retrievement.data_retriever import DataRetriever
    from active_learning_ts.data_retrievement.pool import Pool


class SurrogateModel(PipelineElement, Protocol):
    """
    The goal of a SurrogateModel is to as best as possible, emulate the Data Retrievement process. What constitutes a
    good emulation of the Data Retrievement process may be model/use-case specific
    """


    def __init__(self, config: SurrogateModelConfig) -> None:
        super().__init__(config)

    def post_init(self, data_retriever: DataRetriever):
        self._query_pool = data_retriever.query_pool
        self.query_shape = data_retriever.query_shape
        self.result_shape = data_retriever.result_shape

    def learn(self, points: tf.Tensor, feedback: tf.Tensor):
        """
        Trains the model at the given points using the given feedback

        :param points: the points queried
        :param feedback: the feedback to be used for training
        :return: model specific
        """
        ...


@dataclass(frozen = True)    
class SurrogateModelConfig(BlueprintConfig):
    pipline_element: Type[SurrogateModel]

@dataclass(frozen = True)    
class SurrogateModelData():
    query_candidates: tf.Tensor = tf.convert_to_tensor([])
    actual_queries: tf.Tensor = tf.convert_to_tensor([])
    query_results: tf.Tensor = tf.convert_to_tensor([])