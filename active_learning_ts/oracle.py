from active_learning_ts.instance_properties.instance_objective import InstanceObjective
from active_learning_ts.instance_properties.instance_cost import InstanceCost
from active_learning_ts.data_retrievement.data_retriever import DataRetriever
from active_learning_ts.data_instance import DataInstanceFactory
from active_learning_ts.data_blackboard import Blackboard


class Oracle:
    """
    The Oracle is a wrapper for the Data retrievement process.

    Queries passed to the oracle are queried from the Data retriever.
    Cost, Object, the actual points queried, and the results are the then posted on the Blackboard
    """

    def __init__(
            self,
            data_instance_factory: DataInstanceFactory,
            blackboard: Blackboard,
            data_retriever: DataRetriever,
            instance_cost: InstanceCost,
            instance_level_objective: InstanceObjective,
    ) -> None:
        self.blackboard: Blackboard = blackboard
        self.data_instance_factory: DataInstanceFactory = data_instance_factory
        self.data_retriever: DataRetriever = data_retriever
        self.instance_level_objective: InstanceObjective = instance_level_objective
        self.instance_cost: InstanceCost = instance_cost

    def query(self, query_candidate_indices):
        if len(query_candidate_indices) == 0:
            return

        new_instance = self.data_instance_factory()
        self.blackboard.add_instance(new_instance)

        self.blackboard.last_instance.query_candidates = query_candidate_indices

        actual_queries, query_results = self.data_retriever.query(query_candidate_indices)

        self.blackboard.last_instance.actual_queries = actual_queries
        self.blackboard.last_instance.query_results = query_results

        quality = self.instance_level_objective.apply(query_results)
        self.blackboard.last_instance.quality = quality

        cost = self.instance_cost.apply(actual_queries)
        self.blackboard.last_instance.cost = cost
