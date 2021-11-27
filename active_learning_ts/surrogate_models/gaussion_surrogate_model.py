from typing import List

from active_learning_ts.surrogate_models.surrogate_model import SurrogateModel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic
import tensorflow as tf


class GaussianSurrogateModel(SurrogateModel):
    """
    implements gaussian process
    """

    def __init__(self):
        self.gpr = GaussianProcessRegressor(kernel=RationalQuadratic())
        self.training_points = []
        self.training_values = []

    # TODO: convert everywhere where there is list, to tensor
    def learn(self, points: List[tf.Tensor], values: List[tf.Tensor]):
        self.training_points = self.training_points + points
        self.training_values = self.training_values + values
        self.gpr.fit(self.training_points, self.training_values)

    def uncertainty(self, points: List[tf.Tensor]) -> tf.Tensor:
        y_std = self.gpr.predict(points, return_std=True)[1]
        return tf.convert_to_tensor(y_std)

    def query(self, points: List[tf.Tensor]) -> List[tf.Tensor]:
        y_mean = self.gpr.predict(points)
        return tf.constant(y_mean)
