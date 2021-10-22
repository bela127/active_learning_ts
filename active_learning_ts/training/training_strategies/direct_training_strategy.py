from typing import List

import tensorflow as tf

from active_learning_ts.training.training_strategy import TrainingStrategy


class DirectTrainingStrategy(TrainingStrategy):

    def train(self, query: List[tf.Tensor], feedback: List[tf.Tensor]):
        self.surrogate_model.learn(query, feedback)
