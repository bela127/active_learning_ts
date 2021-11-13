from typing import List, Protocol, Tuple

import tensorflow as tf


class InterpolationStrategy(Protocol):
    def interpolate(self, queries: List[tf.Tensor], queried_points: List[List[tf.Tensor]],
                    query_results: List[List[tf.Tensor]]) -> Tuple[List[tf.Tensor], List[tf.Tensor]]:
        pass
