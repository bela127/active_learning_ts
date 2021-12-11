from active_learning_ts.query_selection.query_optimizer import QueryOptimizer
import tensorflow as tf

from active_learning_ts.query_selection.query_optimizers.generic_max_query_optimizer import GenericMaximumQueryOptimizer


class MaximumImprovementQueryOptimizer(QueryOptimizer):
    """
    Selects the best queries from random points
    """

    def __init__(self,
                 num_tries: int = 1):
        self.generic_query_optimizer = GenericMaximumQueryOptimizer(lambda x: tf.reduce_sum(x), num_tries)

    def optimize_query_candidates(self, num_queries: int = 1):
        return self.generic_query_optimizer.optimize_query_candidates(num_queries)
