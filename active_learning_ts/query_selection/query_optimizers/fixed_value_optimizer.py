import tensorflow as tf

from active_learning_ts.query_selection.query_optimizer import QueryOptimizer


class FixedValueOptimizer(QueryOptimizer):
    """
    Selects the best queries from random points
    """

    def __init__(self, value: float = 0.5, num_tries: int = 10):
        self.num_tries = num_tries
        self.value = value

    def optimize_query_candidates(
            self
    ):
        queries = self.query_sampler.sample(self.num_tries)
        query_values = self.query_sampler.pool.get_elements_with_index(queries)
        b = tf.reshape(self.selection_criteria.score_queries(query_values), (-1,))
        mask = tf.map_fn(lambda x: x > self.value, b, fn_output_signature=tf.bool)
        return tf.boolean_mask(mask=mask, tensor=query_values)
