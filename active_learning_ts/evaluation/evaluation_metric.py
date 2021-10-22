from typing import Protocol

from active_learning_ts.data_blackboard import Blackboard


class EvaluationMetric(Protocol):
    def post_init(self, blackboard: Blackboard, blueprint):
        self.blackboard = blackboard
        self.blueprint = blueprint

    def eval(self):
        pass

    def get_evaluation(self):
        pass
