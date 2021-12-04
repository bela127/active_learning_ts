from typing import List

import tensorflow as tf

from active_learning_ts.data_blackboard import Blackboard
from active_learning_ts.training.training_strategy import TrainingStrategy


class DirectTrainingStrategy(TrainingStrategy):

    def train(self, blackboard: Blackboard):
        self.surrogate_model.learn(blackboard.last_instance.actual_queries, blackboard.last_instance.query_results)
