from typing import List, Tuple

from active_learning_ts.data_pipeline import DataPipeline
from active_learning_ts.data_retrievement.data_source import DataSource
from active_learning_ts.data_retrievement.interpolation.interpolation_strategy import InterpolationStrategy

import tensorflow as tf


class DataRetriever:
    """
    Uses the given retrievement strategy in order to retrieve data from the given data source
    """

    def __init__(
            self,
            data_source: DataSource,
            augmentation_pipeline: DataPipeline,
            interpolation_strategy: InterpolationStrategy
    ) -> None:
        self.data_source = data_source
        self.augmentation_pipeline = augmentation_pipeline
        self.interpolation_strategy = interpolation_strategy
        self.query_pool = None

    def query(self, query_candidates):
        actual_queries, query_results = self.retrieve(
            query_candidates
        )
        augmented_query_results = self.augmentation_pipeline.apply(query_results)
        return actual_queries, augmented_query_results

    def retrieve(self,
                 query_candidates: List[tf.Tensor]) -> Tuple[List[tf.Tensor], List[tf.Tensor]]:
        actual_queries = self.possible_queries(query_candidates)

        out_queries = []
        out_query_results = []
        for query in actual_queries:
            sent_queries, query_results = self.data_source.query(query)
            out_queries.append(sent_queries)
            out_query_results.append(query_results)

        return self.interpolation_strategy.interpolate(query_candidates, out_queries, out_query_results)

    def possible_queries(self, query_candidates):
        return self.data_source.possible_queries().get_elements(query_candidates)

    def get_query_pool(self):
        if self.query_pool is None:
            return self.data_source.possible_queries()
        else:
            return self.query_pool
