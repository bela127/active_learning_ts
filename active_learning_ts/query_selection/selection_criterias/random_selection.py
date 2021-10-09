from typing import List

import tensorflow as tf
from active_learning_ts.query_selection.selection_criteria import SelectionCriteria


class RandomSelection(SelectionCriteria):
    def __init__(self, dim: int, min_x: float = -10.0, max_x: float = 10.0):
        self.min_x = min_x
        self.max_x = max_x
        self.dim = dim

    @tf.function
    def generate_queries(self, num_queries: int = 1) -> List[tf.Tensor]:
        return [tf.random.uniform(shape=(self.dim,), minval=self.min_x, maxval=self.max_x) for _ in
                range(0, num_queries)]
