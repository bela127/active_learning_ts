from typing import Tuple

import tensorflow as tf

from active_learning_ts.logging.data_blackboard import Blackboard
from active_learning_ts.logging.data_instance import DataInstanceFactory
from active_learning_ts.instance_properties.instance_cost import InstanceCost
from active_learning_ts.instance_properties.instance_objective import InstanceObjective
from active_learning_ts.data_retrievement.pool import Pool
from active_learning_ts.pipeline_element import PipelineElement


class Oracle(PipelineElement):
    """
    The Oracle is a wrapper for the Data retrievement process.

    Queries passed to the oracle are queried from the Data retriever.
    Cost, Object, the actual points queried, and the results are the then posted on the Blackboard
    """

    def __init__(
            self,
            data_instance_factory: DataInstanceFactory,
            blackboard: Blackboard,
            data_retriever: PipelineElement,
            instance_cost: InstanceCost,
            instance_level_objective: InstanceObjective,
    ) -> None:
        self.query_shape = data_retriever.query_shape
        self.result_shape = data_retriever.result_shape
        self.blackboard: Blackboard = blackboard
        self.data_instance_factory: DataInstanceFactory = data_instance_factory
        self.data_retriever: PipelineElement = data_retriever
        self.instance_level_objective: InstanceObjective = instance_level_objective
        self.instance_cost: InstanceCost = instance_cost
        self.query_shape = data_retriever.query_shape
        self.result_shape = data_retriever.result_shape

    def query(self, query_candidate_indices) -> Tuple[tf.Tensor, tf.Tensor]:
        new_instance = self.data_instance_factory()
        self.blackboard.add_instance(new_instance)

        self.blackboard.last_instance.query_candidates = query_candidate_indices

        if len(query_candidate_indices) == 0:
            return query_candidate_indices, query_candidate_indices

        actual_queries, query_results = self.data_retriever.query(query_candidate_indices)

        self.blackboard.last_instance.actual_queries = actual_queries
        self.blackboard.last_instance.query_results = query_results

        quality = self.instance_level_objective.apply(query_results)
        self.blackboard.last_instance.quality = quality

        cost = self.instance_cost.apply(actual_queries)
        self.blackboard.last_instance.cost = cost

        return actual_queries, query_results

    @property
    def query_pool(self) -> Pool:
        return self.data_retriever.query_pool
