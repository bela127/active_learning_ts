from typing import Protocol

from active_learning_ts.query_selection.query_sampler import QuerySampler
from active_learning_ts.query_selection.selection_criteria import SelectionCriteria
from active_learning_ts.surrogate_model.surrogate_model import SurrogateModel


class QueryOptimizer(Protocol):
    """
    The query optimizer uses data from the blackboard and the surrogate model in order to generate queries. These
    queries are then scored by the selection Criteria, and then some of them are returned.
    """

    def __init__(self):
        self.surrogate_model: SurrogateModel = None
        self.selection_criteria: SelectionCriteria = None
        self.query_sampler: QuerySampler = None
        self.generic_query_optimizer: QueryOptimizer = None

    def post_init(self, surrogate_model: SurrogateModel,
                  selection_criteria: SelectionCriteria,
                  query_sampler: QuerySampler
                  ):
        self.surrogate_model = surrogate_model
        self.selection_criteria = selection_criteria
        self.query_sampler = query_sampler

        if hasattr(self, 'generic_query_optimizer') and self.generic_query_optimizer is not None:
            self.generic_query_optimizer.post_init(surrogate_model, selection_criteria, query_sampler)

    def optimize_query_candidates(self):
        pass
