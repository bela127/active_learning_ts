from __future__ import annotations
from typing import TYPE_CHECKING

from typing import Protocol

import tensorflow as tf

if TYPE_CHECKING:
    from typing import Tuple
    
    from active_learning_ts.experiments.blueprint_config import BlueprintConfig
    from active_learning_ts.data_retrievement.pool import Pool


# TODO make datasource implement this
class PipelineElement(Protocol):
    query_shape: Tuple
    result_shape: Tuple
    _query_pool: Pool

    def __init__(self, config: BlueprintConfig) -> None:
        super().__init__()


    def query(self, query: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Creates a result for a given query
        :param query: [nr_of_queries, query]
        :return: ([nr_of_queries, actual_query], [nr_of_queries, query_result])
        """
        ...

    @property
    def query_pool(self) -> Pool:
        return self._query_pool

class UncertaintyPipelineElement(PipelineElement, Protocol):

    def uncertainty(self, query: tf.Tensor) -> tf.Tensor:
        """
        Returns the uncertainty of the model at the given points. A higher number means the model is less certain
        :param query: [nr_of_queries, query], the queries at which the uncertainty should be measured
        :return: [nr_of_queries, query_result], the uncertainties for every query.
        """
        ...
