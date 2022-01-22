from typing import List

import tensorflow as tf

from active_learning_ts.pool import Pool
from active_learning_ts.pools.retrievement_strategy import RetrievementStrategy


class DiscreteVectorPool(Pool):
    def __init__(self, in_dim: int, queries: [tf.Tensor], find_streategy: RetrievementStrategy):
        self.queries = queries
        self.shape = (in_dim,)
        self.ranges = []
        self.find_strategy = find_streategy

        for i in range(in_dim):
            minimum = queries[0][i]
            maximum = queries[0][i]
            for j in range(1, len(queries)):
                minimum = min(minimum, queries[j][i])
                maximum = max(maximum, queries[j][i])

            self.ranges.append([(minimum, maximum)])

    def get_elements(self, elements: tf.Tensor) -> tf.Tensor:
        return self.find_strategy.find(elements)

    def get_element_normalized(self, element: tf.Tensor) -> tf.Tensor:
        indices = tf.unstack(element)

        query = []
        for i in range(len(indices)):
            lower, upper = self.ranges[i][0]
            size = upper - lower

            query.append((indices[i] * size) + lower)

        query = tf.convert_to_tensor([tf.stack(query)])
        return self.get_elements(query)[0]

    @tf.function
    def _normalize(self, query_candidate):
        indices = tf.unstack(query_candidate)

        query = []
        for i in range(len(indices)):
            lower, upper = self.ranges[i][0]
            size = tf.cond(lower == upper, lambda: 1.0, lambda: upper - lower)

            query.append((indices[i] - lower) / size)

        return tf.stack(query)

    def get_all_elements(self) -> List[tf.Tensor]:
        return self.queries[:]

    def is_discrete(self) -> bool:
        return True

    def get_elements_with_index(self, indices: tf.Tensor):
        if indices.dtype == tf.dtypes.int32:
            return tf.gather(self.queries, indices)
        else:
            return indices
