import tensorflow as tf

from active_learning_ts.query_selection.query_sampler import QuerySampler


class NoQuerySampler(QuerySampler):
    def sample(self, num_queries: int = 1) -> tf.Tensor:
        pass

    def update_pool(self, pool):
        pass
