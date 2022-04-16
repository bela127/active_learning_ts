from typing import Protocol, Tuple

import tensorflow as tf


class InterpolationStrategy(Protocol):
    """
    A single point that is selected by the query optimizer, can equate to multiple queries from the data source.
    The job of the interpolation strategy is to interpolate the results of those queries
    """
    def interpolate(self, queries: tf.Tensor, queried_points: tf.Tensor,
                    query_results: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Interpolates the query results and queried points.

        :param: queries, a list of tensors, that were selected to be queried from the data source
        :param: queried_points, a list of lists of tensors that were actually queried from the data source. Each list
            corresponds to the respective point from queries
        :param: query_results, a list of lists of the results of the respective queries from queried_points.
        """
        pass
