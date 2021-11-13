from typing import List, Protocol

import tensorflow as tf


class InterpolationStrategy(Protocol):
    def interpolate(self, query: tf.Tensor, queried_points: List[tf.Tensor],
                    query_results: List[tf.Tensor]) -> List[tf.Tensor]:
        pass
