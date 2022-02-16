import tensorflow as tf

from active_learning_ts.query_selection.query_sampler import QuerySampler


class RandomContinuousQuerySampler(QuerySampler):
    def sample(self, num_queries: int = 1) -> tf.Tensor:
        if self.pool.is_discrete():
            elems = self.pool.get_all_elements()
            return tf.random.uniform((num_queries,), 0, len(elems), tf.dtypes.int32)
        else:
            a = self.pool.get_elements_normalized(tf.random.uniform(shape=(num_queries, self.pool.shape[0])))
            a = tf.reshape(a, (a.shape[0] * a.shape[1], a.shape[2]))
        return tf.convert_to_tensor(a)

    def update_pool(self, pool):
        self.pool = pool
