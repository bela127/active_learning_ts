from typing import List

from active_learning_ts.pools.retrievement_strategy import RetrievementStrategy
from active_learning_ts.surrogate_models.surrogate_model import SurrogateModel
import tensorflow as tf


class NoSurrogateModel(SurrogateModel):
    def __init__(self, retrievement_strategy: RetrievementStrategy):
        self.retrievement_strategy = retrievement_strategy

    def uncertainty(self, points: List[tf.Tensor]) -> tf.Tensor:
        return tf.convert_to_tensor([.0] * len(points))

    def query(self, points: List[tf.Tensor]) -> List[tf.Tensor]:
        return [x for sublist in self.retrievement_strategy.find(points) for x in sublist]

