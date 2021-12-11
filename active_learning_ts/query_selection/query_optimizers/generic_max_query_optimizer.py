from active_learning_ts.query_selection.query_optimizer import QueryOptimizer
import tensorflow as tf


class GenericMaximumQueryOptimizer(QueryOptimizer):
    """
    Selects the best queries from random points
    """

    def __init__(self,
                 function,
                 num_tries: int = 1):
        self.num_tries = num_tries
        self.function = function

    def optimize_query_candidates(
            self, num_queries: int = 1
    ):
        out = []

        for i in range(0, num_queries):
            a = self.query_sampler.sample(self.num_tries)
            b = self.selection_criteria.score_queries(a)
            b = tf.map_fn(self.function, b)
            best = tf.argmax(b)
            out.append(a[best])
        return tf.convert_to_tensor(out, dtype=tf.dtypes.float32)
