from active_learning_ts.logging.data_blackboard import Blackboard
from active_learning_ts.training.training_strategy import TrainingStrategy


class Trainer:
    """
    The trainer is responsible for training the given Surrogate Model.

    The given Training strategy is used to train the Model.
    """
    def __init__(self,
                 blackboard: Blackboard,
                 training_strategy: TrainingStrategy):
        self.training_strategy = training_strategy
        self.blackboard = blackboard

    def train(self):
        self.training_strategy.train(self.blackboard)
