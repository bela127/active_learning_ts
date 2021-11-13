from typing import List, Tuple

import tensorflow as tf

from active_learning_ts.data_retrievement.interpolation.interpolation_strategy import InterpolationStrategy


class FlatMapInterpolation(InterpolationStrategy):

    def interpolate(self, queries: List[tf.Tensor], queried_points: List[List[tf.Tensor]],
                    query_results: List[List[tf.Tensor]]) -> Tuple[List[tf.Tensor], List[tf.Tensor]]:
        out_queries = [x for sublist in queried_points for x in sublist]
        out_results = [x for sublist in query_results for x in sublist]

        return out_queries, out_results
