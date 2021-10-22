from typing import Protocol, List
import tensorflow as tf

from active_learning_ts.surrogate_models.surrogate_model import SurrogateModel


class TrainingStrategy(Protocol):

    def train(self, query: List[tf.Tensor], feedback: List[tf.Tensor]):
        pass

    def post_init(self, surrogate_model: SurrogateModel):
        self.surrogate_model = surrogate_model
