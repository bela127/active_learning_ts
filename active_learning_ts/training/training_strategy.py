from typing import Protocol

from active_learning_ts.logging.data_blackboard import Blackboard
from active_learning_ts.surrogate_model.surrogate_model import SurrogateModel


class TrainingStrategy(Protocol):
    """
    A training Strategy is responsible for giving Feedback to the surrogate model. Training Strategies may be specific
    to specific SurrogateModels

    Given data from the blackboard, the Training strategy creates feedback for the surrogate model, so that it may
    train, or it may train the surrogate model itself.
    """

    def train(self, blackboard: Blackboard, surrogate_blackboard: Blackboard):
        pass

    def post_init(self, surrogate_model: SurrogateModel):
        self.surrogate_model = surrogate_model
