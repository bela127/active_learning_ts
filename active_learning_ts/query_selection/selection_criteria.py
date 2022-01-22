from typing import Protocol, List

import tensorflow as tf

from active_learning_ts.instance_properties.instance_cost import InstanceCost
from active_learning_ts.instance_properties.instance_objective import InstanceObjective
from active_learning_ts.knowledge_discovery.knowledge_discovery_task import KnowledgeDiscoveryTask
from active_learning_ts.surrogate_models.surrogate_model import SurrogateModel


class SelectionCriteria(Protocol):
    """
    Selection Criteria aims to score any given point, on how good of a query it would be. This is used in order to, with
    the specific task in mind, be able to select the best query of a bunch of different queries
    """

    def __init__(self):
        self.surrogate_model: SurrogateModel = None
        self.knowledge_discovery: KnowledgeDiscoveryTask = None
        self.selection_criteria: List[SelectionCriteria] = None
        self.instance_cost: InstanceCost = None
        self.instance_objective: InstanceObjective = None

    def score_queries(self, queries: tf.Tensor) -> tf.Tensor:
        """
        Scores the given queries on how good they are. The higher the number the better
        :param queries: the points to be scored
        :return: a 2D tensor of their respective scores
        """
        pass

    def post_init(self, surrogate_model: SurrogateModel, knowledge_discovery: KnowledgeDiscoveryTask,
                  instance_cost: InstanceCost = None, instance_objective: InstanceObjective = None):
        self.surrogate_model = surrogate_model
        self.knowledge_discovery = knowledge_discovery
        self.instance_cost = instance_cost
        self.instance_objective = instance_objective

        if hasattr(self, 'selection_criteria') and self.selection_criteria is not None:
            for x in self.selection_criteria:
                x.post_init(surrogate_model, knowledge_discovery, instance_cost, instance_objective)
