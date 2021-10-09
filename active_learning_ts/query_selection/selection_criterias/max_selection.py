from typing import List

import tensorflow as tf
from active_learning_ts.query_selection.selection_criteria import SelectionCriteria
from active_learning_ts.query_selection.selection_criterias.random_selection import RandomSelection


class MaxSelection(SelectionCriteria):
    def __init__(self, dim: int, min_x: float = -10.0, max_x: float = 10.0, exploration_radius: float = .1):
        self.min_x = min_x
        self.max_x = max_x
        self.dim = dim
        self.exploration_radius = exploration_radius * (max_x - min_x)
        self.current_max = None

    @tf.function
    def _generate_query(self):
        explorer = tf.random.uniform(shape=(self.dim,), minval=self.exploration_radius, maxval=self.exploration_radius)
        out = self.current_max + explorer
        return tf.map_fn(lambda x: self.min_x if x < self.min_x else x,
                         tf.map_fn(lambda x: self.min_x if x > self.max_x else x, out))

    @tf.function
    def generate_queries(self, num_queries: int = 1) -> List[tf.Tensor]:
        if self.current_max is None:
            return RandomSelector(dim=self.dim, min_x=self.min_x, max_x=self.max_x).generate_queries(num_queries)
        return [self._generate_query() for _ in range(0, num_queries)]

    @tf.function
    def inform(self, queries: List[tf.Tensor], query_results: List[tf.Tensor]) -> None:
        ind_max = tf.argmax(tf.stack([tf.norm(x) for x in query_results]))
        # should work to index array in tf.function. normally you cannot do that in graph mode
        self.current_max = tf.py_function(lambda x: queries[int(ind_max)], inp=[ind_max], Tout=[tf.Tensor])
