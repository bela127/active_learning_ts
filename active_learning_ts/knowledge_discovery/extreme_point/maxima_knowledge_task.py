import numpy
import tensorflow as tf
from scipy import optimize

from active_learning_ts.knowledge_discovery.knowledge_discovery_task import KnowledgeDiscoveryTask
from active_learning_ts.query_selection.query_sampler import QuerySampler
from active_learning_ts.surrogate_models.surrogate_model import SurrogateModel


class MaximaKnowledgeDiscoveryTask(KnowledgeDiscoveryTask):
    def post_init(self, surrogate_model: SurrogateModel, sampler: QuerySampler):
        super(MaximaKnowledgeDiscoveryTask, self).post_init(surrogate_model, sampler)
        if not (len(surrogate_model.point_shape) == 1 and len(surrogate_model.value_shape) == 1 and
                surrogate_model.value_shape[0] == 1):
            raise AttributeError('This class only works with  multivariate functions')

    def f(self, x):
        zero = tf.constant([.0] * len(x))
        one = tf.constant([1.0] * len(x))
        x = tf.convert_to_tensor(x, dtype=tf.dtypes.float32)

        if not (tf.reduce_all(x < one) and tf.reduce_all(zero < x)):
            return -numpy.Inf

        x = self.surrogate_model.get_query_pool().get_element_normalized(x)
        x, y = self.surrogate_model.query(x)
        return -y[0][0]

    def learn(self, num_queries):
        return tf.convert_to_tensor(
            optimize.fmin(lambda x: self.f(x), tf.fill(self.surrogate_model.point_shape, .5))[0])

    def uncertainty(self, points: tf.Tensor) -> tf.Tensor:
        return tf.fill(points[0].shape, .0)
