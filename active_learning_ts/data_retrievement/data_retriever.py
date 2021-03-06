from typing import Tuple

import tensorflow as tf

from active_learning_ts.data_pipeline import DataPipeline
from active_learning_ts.data_retrievement.data_source import DataSource
from active_learning_ts.data_retrievement.interpolation_strategy import InterpolationStrategy
from active_learning_ts.data_retrievement.pool import Pool
from active_learning_ts.queryable import Queryable


class DataRetriever(Queryable):
    """
    Uses the given retrievement strategy in order to retrieve data from the given data source
    """

    def __init__(
            self,
            data_source: DataSource,
            augmentation_pipeline: DataPipeline,
            interpolation_strategy: InterpolationStrategy
    ) -> None:
        self.point_shape = data_source.point_shape
        self.value_shape = data_source.value_shape
        self.data_source = data_source
        self.augmentation_pipeline = augmentation_pipeline
        self.interpolation_strategy = interpolation_strategy
        self.query_pool: Pool

    def query(self, query_candidates) -> Tuple[tf.Tensor, tf.Tensor]:
        actual_queries, query_results = self.retrieve(
            query_candidates
        )
        augmented_query_results = self.augmentation_pipeline.apply(query_results)
        return actual_queries, augmented_query_results

    def retrieve(self,
                 query_candidates: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        actual_queries = self.possible_queries(query_candidates)

        out_queries = []
        out_query_results = []
        for query in actual_queries:
            sent_queries, query_results = self.data_source.query(query)
            out_queries.append(sent_queries)
            out_query_results.append(query_results)

        query_candidates = tf.convert_to_tensor(query_candidates)
        out_queries = tf.convert_to_tensor(out_queries)
        out_query_results = tf.convert_to_tensor(out_query_results)
        return self.interpolation_strategy.interpolate(query_candidates, out_queries, out_query_results)

    def possible_queries(self, query_candidates):
        return self.data_source.possible_queries().get_elements(query_candidates)

    def get_query_pool(self) -> Pool:
        return self.data_source.possible_queries()
