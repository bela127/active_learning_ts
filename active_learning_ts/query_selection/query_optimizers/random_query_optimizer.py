import random
from typing import List

from active_learning_ts.query_selection.selection_criteria import SelectionCriteria
from active_learning_ts.query_selection.query_optimizer import QueryOptimizer
import tensorflow as tf


class RandomQueryOptimizer(QueryOptimizer):
    # TODO: should probably take shape instead of dim here, might want to work with matrix functions as well, this goes
    #  for all places where dim is an int, not sure if the gaussian library supports that
    def __init__(self, dim: int, min_x: float = -10.0, max_x: float = 10.0,
                 num_tries: int = 1):
        self.min_x = min_x
        self.max_x = max_x
        self.dim = dim
        self.num_tries = num_tries

    def optimize_query_candidates(
            self, num_queries: int = 1, possible_queries: List[tf.Tensor] = None
    ):
        out = []
        if possible_queries is None:
            for i in range(0, num_queries):
                a = [tf.random.uniform(shape=(self.dim,), minval=self.min_x, maxval=self.max_x) for _ in
                     range(0, self.num_tries)]
                b = self.selection_criteria.score_queries(a)
                best = tf.argmax(b)
                out.append(a[best])
            return out
        return possible_queries[random.randint(0, len(possible_queries))]
