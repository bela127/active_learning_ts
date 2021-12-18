from typing import List, Tuple

import tensorflow as tf

from active_learning_ts.data_retrievement.interpolation.interpolation_strategy import InterpolationStrategy


class FlatMapInterpolation(InterpolationStrategy):

    def interpolate(self, queries: tf.Tensor, queried_points: tf.Tensor,
                    query_results: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """

        :param queries: 2D tensor
        :param queried_points: 3D Tensor
        :param query_results: 3D Tensor
        :return: Tuple of 2D Tensor
        """

        queries_shape = queried_points.shape
        out_queries = tf.reshape(queried_points, (-1, queries_shape[2]))

        results_shape = query_results.shape
        out_results = tf.reshape(query_results, (results_shape[1] * results_shape[0], results_shape[2]))

        return out_queries, out_results
