from typing import Protocol

from active_learning_ts.data_blackboard import Blackboard


class EvaluationMetric(Protocol):
    """
    Considering all parts of the experiment, (passed in as blueprint), measures certain metrics about the experiment,
    such as total number of elapsed rounds, average/total training time, uncertainty improvement per round...
    """

    def post_init(self, blackboard: Blackboard, blueprint):
        self.blackboard = blackboard
        self.blueprint = blueprint

    def eval(self) -> None:
        """
        Evaluates the metric after the current round
        :return: None
        """
        pass

    def get_evaluation(self):
        """
        Returns all evaluations as a list, or number when using aggregate metrics
        :return: all evaluations as a number or list
        """
        pass
