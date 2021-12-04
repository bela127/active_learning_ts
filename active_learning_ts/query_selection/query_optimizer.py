from typing import Protocol, List, Union
import tensorflow as tf

from active_learning_ts.query_selection.query_sampler import QuerySampler
from active_learning_ts.query_selection.selection_criteria import SelectionCriteria
from active_learning_ts.surrogate_models.surrogate_model import SurrogateModel


class QueryOptimizer(Protocol):
    """
    The query optimizer uses data from the blackboard and the surrogate model in order to generate queries. These
    queries are then scored by the selection Criteria, and then some of them are returned.
    """

    def post_init(self, surrogate_model: SurrogateModel,
                  selection_criteria: SelectionCriteria,
                  query_sampler: QuerySampler
                  ):
        self.surrogate_model = surrogate_model
        self.selection_criteria = selection_criteria
        self.query_sampler = query_sampler

    # TODO: selectionCriteria should also consider cost
    def optimize_query_candidates(
            self, num_queries: int = 1
    ):
        pass
