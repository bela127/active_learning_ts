import tensorflow as tf

from active_learning_ts.query_selection.query_optimizer import QueryOptimizer


class GenericMaximumQueryOptimizer(QueryOptimizer):
    """
    Selects the best queries from random points
    """

    def __init__(self,
                 function,
                 num_tries: int = 1,
                 num_queries: int = 1):
        self.num_tries = num_tries
        self.function = function
        self.num_queries = num_queries

    def optimize_query_candidates(
            self
    ):
        total_queries = self.num_tries * self.num_queries
        queries = self.query_sampler.sample(total_queries)
        query_values = self.query_sampler.pool.get_elements_with_index(queries)
        b = self.selection_criteria.score_queries(query_values)
        b = tf.map_fn(self.function, b)
        return tf.gather(queries, tf.math.top_k(b, self.num_queries).indices)

