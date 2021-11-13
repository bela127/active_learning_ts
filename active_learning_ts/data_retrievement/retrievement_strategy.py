from active_learning_ts.data_retrievement.interpolation.interpolation_strategy import InterpolationStrategy
from active_learning_ts.pool import Pool
from typing import Protocol, Union, List, Tuple
from active_learning_ts.data_retrievement.data_source import DataSource
import tensorflow as tf


class RetrievementStrategy(Protocol):
    """
    Is responsible for retrieving given queries from the given Data source.
    """
    query_pool: Union[Pool, None] = None

    def __init__(self) -> None:
        self.data_source_pool: Pool = None
        self.query_pool: Pool = None
        self.interpolation_strategy: InterpolationStrategy = None

    def post_init(self, query_pool: Pool, interpolation_strategy: InterpolationStrategy):
        self.data_source_pool = query_pool
        self.interpolation_strategy = interpolation_strategy

    def retrieve(self, data_source: DataSource,
                 query_candidates: List[tf.Tensor]) -> Tuple[List[tf.Tensor], List[tf.Tensor]]:
        actual_queries = self.possible_queries(query_candidates)

        out_queries = []
        out_query_results = []
        for query in actual_queries:
            sent_queries, query_results = data_source.query(query)
            out_queries.append(sent_queries)
            out_query_results.append(query_results)

        return self.interpolation_strategy.interpolate(query_candidates, out_queries, out_query_results)

    def possible_queries(self, query_candidates):
        return self.data_source_pool.get_elements(query_candidates)

    def get_query_pool(self):
        if self.query_pool is None:
            return self.data_source_pool
        else:
            return self.query_pool
