from active_learning_ts.query_selection.query_sampler import QuerySampler
import tensorflow as tf


class RandomContinuousQuerySampler(QuerySampler):
    def sample(self, num_queries: int = 1) -> tf.Tensor:
        # TODO change this to tensor op
        if self.pool.is_discrete():
            elems = self.pool.get_all_elements()
            return tf.random.uniform((num_queries,), 0, len(elems), tf.dtypes.int32)
        else:
            a = self.pool.get_elements_normalized(
                [tf.random.uniform(shape=self.pool.shape) for _ in
                 range(0, num_queries)])
            a = [val for sublist in a for val in sublist]
        return tf.convert_to_tensor(a)
