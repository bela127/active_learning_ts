from typing import List, Protocol, Tuple

import tensorflow as tf


class InterpolationStrategy(Protocol):
    def interpolate(self, queries: tf.Tensor, queried_points: tf.Tensor,
                    query_results: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        pass
