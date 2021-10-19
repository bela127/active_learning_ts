from typing import Protocol, List
import tensorflow as tf


class QueryOptimizer(Protocol):
    # TODO, implement possible queries
    # TODO: the retrievement Strategy should not be passed here. That is the responsibility of DataSource. However,
    #  knowing that the exact data point can be queried can be used to implement a more efficient retrievement Strategy

    # I am not sure the optimizer should know anything about the retrievement Strategy: Matthew: neither do I, i am
    # saying it should, but it should know the possible queries

    # TODO: query optimizer should also consider cost
    # TODO cost should be returned here. in the case that the cost function is not easy to calculate, we should avoid
    #  calculating it multiple times. It is currently also caluclated in the oracle

    # the oracle calculates the real cost, while here the expected cost of possible candidates is calculated,
    # so it has to be calculated again for each instance
    def optimize_query_candidates(
            self, num_queries: int = 1, possible_queries: List[tf.Tensor] = None
    ):
        pass
