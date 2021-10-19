from typing import List

from active_learning_ts.surrogate_models.surrogate_model import SurrogateModel
import tensorflow as tf

from active_learning_ts.training.training_strategy import TrainingStrategy


class DirectTrainingStrategy(TrainingStrategy):
    def __init__(self, surrogate_model: SurrogateModel):
        super().__init__()
        self.surrogate_model = surrogate_model

    def train(self, query: List[tf.Tensor], feedback: List[tf.Tensor]):
        self.surrogate_model.learn(query, feedback)
