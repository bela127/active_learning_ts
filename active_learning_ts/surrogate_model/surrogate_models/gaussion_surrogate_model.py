from typing import Tuple

import tensorflow as tf
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic

from active_learning_ts.data_retrievement.pool import Pool
from active_learning_ts.data_retrievement.pools.continuous_vector_pool import ContinuousVectorPool
from active_learning_ts.surrogate_model.surrogate_model import SurrogateModel


class GaussianSurrogateModel(SurrogateModel):
    """
    implements gaussian process
    """

    def __init__(self):
        self.gpr = GaussianProcessRegressor(kernel=RationalQuadratic())
        self.training_points = None
        self.training_values = None

    def learn(self, points: tf.Tensor, values: tf.Tensor):
        if self.training_points is None:
            self.training_points = points
            self.training_values = values
        else:
            self.training_points = tf.concat([self.training_points, points], 0)
            self.training_values = tf.concat([self.training_values, values], 0)

        self.gpr.fit(self.training_points, self.training_values)

    def uncertainty(self, points: tf.Tensor) -> tf.Tensor:
        y_std = self.gpr.predict(points, return_std=True)[1]
        return tf.convert_to_tensor(y_std, dtype=tf.dtypes.float32)

    def query(self, points: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        y_mean = self.gpr.predict(points)
        return points, tf.convert_to_tensor(y_mean, dtype=tf.dtypes.float32)

    def get_query_pool(self) -> Pool:
        if not self.query_pool.is_discrete():
            return self.query_pool
        else:
            self.query_pool = ContinuousVectorPool(self.query_pool.shape[0], ranges=self.query_pool.get_ranges())
            return self.query_pool
