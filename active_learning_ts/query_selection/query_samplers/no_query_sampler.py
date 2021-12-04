from active_learning_ts.query_selection.query_sampler import QuerySampler
import tensorflow as tf


class NoQuerySampler(QuerySampler):
    def sample(self, num_queries: int = 1) -> tf.Tensor:
        pass
