from typing import Protocol


class QueryOptimizer(Protocol):
    # TODO: change SurrogateModel back to Oracle after fully implementing oracle
    # TODO, implement possible_queries
    def optimize_query_candidates(
            self,  num_queries: int = 1
    ):
        pass
