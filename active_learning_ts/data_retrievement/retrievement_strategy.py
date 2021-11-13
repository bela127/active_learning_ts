from active_learning_ts.pool import Pool
from typing import Protocol, Union, List, Tuple
from active_learning_ts.data_retrievement.data_source import DataSource
import tensorflow as tf


class RetrievementStrategy(Protocol):
    """
    Is responsible for retrieving given queries from the given Data source.
    """
    query_pool: Union[Pool, None] = None

    def __init__(self, query_pool: Union[Pool, None] = None) -> None:
        self.query_pool = query_pool

    def retrieve(self, data_source: DataSource,
                 query_candidates: List[tf.Tensor]) -> Tuple[List[tf.Tensor], List[tf.Tensor]]:
        actual_queries = self.possible_queries(query_candidates)

        out_queries = []
        out_query_results = []
        for query in actual_queries:
            sent_queries, query_results = data_source.query(query)
            # TODO: implement interpolation and interpolate results here, for now, flatmapping is enough
            out_queries += sent_queries
            out_query_results += query_results

        return out_queries, out_query_results

    def possible_queries(self, query_candidates):
        return self.data_source_pool.get_elements(query_candidates)

    def post_init(self, data_source_pool: Pool):
        self.data_source_pool = data_source_pool

    def get_query_pool(self):
        # TODO this is temporary because query pool is not yet implemented
        if self.query_pool is None:
            return self.data_source_pool
        else:
            return self.query_pool
