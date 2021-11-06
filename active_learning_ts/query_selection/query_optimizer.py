from typing import Protocol, List, Union
import tensorflow as tf

from active_learning_ts.pool import Pool
from active_learning_ts.pools.continuous_vector_pool import ContinuousVectorPool
from active_learning_ts.query_selection.selection_criteria import SelectionCriteria
from active_learning_ts.surrogate_models.surrogate_model import SurrogateModel


class QueryOptimizer(Protocol):
    """
    The query optimizer uses data from the blackboard and the surrogote model in order to generate queries. These
    queries are then scored by the selection Criteria, and then some of them are returned.
    """

    def post_init(self, surrogate_model: SurrogateModel, selection_criteria: SelectionCriteria,
                  query_selection_pool: Pool):
        self.surrogate_model = surrogate_model
        self.selection_criteria = selection_criteria
        self.query_selection_pool = query_selection_pool

    # TODO: selectionCriteria should also consider cost
    def optimize_query_candidates(
            self, num_queries: int = 1, possible_queries: List[tf.Tensor] = None
    ):
        pass
