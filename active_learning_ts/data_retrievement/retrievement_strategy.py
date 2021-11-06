from active_learning_ts.pool import Pool
from typing import Protocol, Union
from active_learning_ts.data_retrievement.data_source import DataSource


class RetrievementStrategy(Protocol):
    """
    Is responsible for retrieving given queries from the given Data source.
    """
    query_pool: Union[Pool, None] = None

    def __init__(self, query_pool: Union[Pool, None] = None) -> None:
        self.query_pool = query_pool

    def retrieve(self, data_source: DataSource, query_candidates):
        actual_queries = self.possible_queries(query_candidates)
        actual_queries, query_results = data_source.query(actual_queries)
        return actual_queries, query_results

    def possible_queries(self, query_candidates):
        return query_candidates

    def post_init(self, data_source_pool: Pool):
        self.data_source_pool = data_source_pool

    def get_query_pool(self):
        if self.query_pool is None:
            return self.data_source_pool
        else:
            return self.query_pool
