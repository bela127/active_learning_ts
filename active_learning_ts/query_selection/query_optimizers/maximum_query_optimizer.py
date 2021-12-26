import tensorflow as tf

from active_learning_ts.query_selection.query_optimizer import QueryOptimizer
from active_learning_ts.query_selection.query_optimizers.generic_max_query_optimizer import GenericMaximumQueryOptimizer


class MaximumQueryOptimizer(QueryOptimizer):
    """
    Selects the best queries from random points
    """

    def __init__(self,
                 num_tries: int = 1):
        self.generic_query_optimizer = GenericMaximumQueryOptimizer(lambda x: tf.reduce_max(x), num_tries)

    def optimize_query_candidates(self):
        return self.generic_query_optimizer.optimize_query_candidates()
