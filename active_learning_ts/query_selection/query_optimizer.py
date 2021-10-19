from typing import Protocol


class QueryOptimizer(Protocol):
    # TODO, implement possible queries
    # TODO: the retrievement Strategy should not be passed here. That is the responsibility of DataSource. However,
    #  knowing that the exact data point can be queried can be used to implement a more efficient retrievement Strategy
    # TODO: query optimizer should also consider cost
    # TODO cost should be returned here. in the case that the cost function is not easy to calculate, we should avoid
    #  calculating it multiple times. It is currently also caluclated in the oracle
    def optimize_query_candidates(
            self,  num_queries: int = 1
    ):
        pass
